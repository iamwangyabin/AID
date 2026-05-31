from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm


@dataclass
class Prediction:
    scores: torch.Tensor
    labels: torch.Tensor
    logits: Optional[torch.Tensor] = None


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def autocast_context(device, amp_dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_dtype is not None,
    )


def binary_scores_from_logits(logits):
    if logits.ndim > 1 and logits.shape[1] > 1:
        return F.softmax(logits, dim=1)[:, 1]
    return logits.sigmoid().flatten()


def load_weights(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def compute_binary_metrics(y_true, y_pred, thres=0.5):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_true = y_true % 2
    y_label = y_pred > thres

    real_mask = y_true == 0
    fake_mask = y_true == 1
    r_acc = accuracy_score(y_true[real_mask], y_label[real_mask]) if real_mask.any() else float("nan")
    f_acc = accuracy_score(y_true[fake_mask], y_label[fake_mask]) if fake_mask.any() else float("nan")
    acc = accuracy_score(y_true, y_label)
    auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
    f1 = f1_score(y_true, y_label, zero_division=0)
    ap = average_precision_score(y_true, y_pred) if fake_mask.any() else 0.0
    return {
        "ap": ap,
        "auc": auc,
        "f1": f1,
        "r_acc0": r_acc,
        "f_acc0": f_acc,
        "acc0": acc,
        "num_real": int((y_true == 0).sum()),
        "num_fake": int((y_true == 1).sum()),
    }


class BinaryLogitPredictor:
    requires_grad = False

    def setup(self, model, device):
        return None

    def predict(self, model, batch, device):
        x, y = batch
        output = model(x)
        if "logits" not in output:
            raise ValueError(f"{model.__class__.__name__} output does not contain 'logits'.")
        logits = output["logits"]
        return Prediction(scores=binary_scores_from_logits(logits), labels=y, logits=logits)


class SoftmaxPredictor(BinaryLogitPredictor):
    def predict(self, model, batch, device):
        x, y = batch
        logits = model(x)["logits"]
        return Prediction(scores=F.softmax(logits, dim=1)[:, 1], labels=y, logits=logits)


class PoundNetPredictor(BinaryLogitPredictor):
    def predict(self, model, batch, device):
        x, y = batch
        logits = model.forward_binary(x)["logits"]
        return Prediction(scores=F.softmax(logits, dim=1)[:, 1], labels=y, logits=logits)


class LNPPredictor(BinaryLogitPredictor):
    def setup(self, model, device):
        from collections import OrderedDict

        from networks.LNP.denoising_rgb import DenoiseNet

        self.model_restoration = DenoiseNet().to(device)
        checkpoint = load_weights("networks/weights/sidd_rgb.pth", map_location=device)
        try:
            self.model_restoration.load_state_dict(checkpoint["state_dict"])
        except RuntimeError:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
            self.model_restoration.load_state_dict(new_state_dict)
        self.model_restoration.eval()

    def predict(self, model, batch, device):
        x, y = batch
        rgb_restored = self.model_restoration(x.float())
        rgb_restored = torch.round(torch.clamp(rgb_restored, 0, 1) * 255.0) / 255.0

        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        rgb_restored = (rgb_restored - mean) / std

        logits = model(rgb_restored)["logits"]
        return Prediction(scores=binary_scores_from_logits(logits), labels=y, logits=logits)


class LGradPredictor(BinaryLogitPredictor):
    requires_grad = True

    def setup(self, model, device):
        from networks.LGrad import build_model

        self.gen_model = build_model(
            gan_type="stylegan",
            module="discriminator",
            resolution=256,
            label_size=0,
            image_channels=3,
        ).to(device)
        state_dict = load_weights(
            "networks/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth",
            map_location=device,
        )
        self.gen_model.load_state_dict(state_dict, strict=True)
        self.gen_model.eval()

    def predict(self, model, batch, device):
        x, y = batch
        input_img = x.float().detach().requires_grad_(True)
        pre = self.gen_model(input_img)
        self.gen_model.zero_grad()
        grads = torch.autograd.grad(
            pre.sum(),
            input_img,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        b_min = torch.min(grads.view(grads.size(0), -1), dim=1)[0]
        grads = grads - b_min.view(-1, 1, 1, 1)
        b_max = torch.max(grads.view(grads.size(0), -1), dim=1)[0]
        grads = grads / b_max.clamp_min(1e-12).view(-1, 1, 1, 1)
        grads = grads * 255.0
        grads = F.interpolate(grads, size=(224, 224), mode="bilinear", align_corners=False)

        mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized_grads = (torch.clamp(grads, 0, 255) / 255.0 - mean) / std
        with torch.no_grad():
            logits = model(normalized_grads)["logits"]
        return Prediction(scores=binary_scores_from_logits(logits), labels=y, logits=logits)


class DIREPredictor(BinaryLogitPredictor):
    def setup(self, model, device):
        from networks.DIRE.script_util import create_model_and_diffusion

        defaults = dict(
            attention_resolutions="32,16,8",
            class_cond=False,
            diffusion_steps=1000,
            dropout=0.1,
            image_size=256,
            learn_sigma=True,
            noise_schedule="linear",
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_fp16=device.type == "cuda",
            use_scale_shift_norm=True,
            timestep_respacing="ddim20",
            channel_mult="",
            num_heads=4,
            num_heads_upsample=-1,
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            use_checkpoint=False,
            use_new_attention_order=False,
        )
        self.diffusion_model, self.diffusion = create_model_and_diffusion(**defaults)
        state_dict = load_weights("weights/256x256_diffusion_uncond.pt", map_location="cpu")
        self.diffusion_model.load_state_dict(state_dict)
        self.diffusion_model.to(device)
        if device.type == "cuda":
            self.diffusion_model.convert_to_fp16()
        self.diffusion_model.eval()
        self.reverse_fn = self.diffusion.ddim_reverse_sample_loop
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, model, batch, device):
        import cv2

        x, y = batch
        latent = self.reverse_fn(self.diffusion_model, x.shape, noise=x.float(), clip_denoised=True, real_step=0)
        recons = self.diffusion.ddim_sample_loop(
            self.diffusion_model,
            latent.shape,
            noise=latent,
            clip_denoised=True,
            real_step=0,
        )
        dire = torch.abs(x - recons)
        dire = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

        transformed_batch = []
        for image in dire:
            retval, buffer = cv2.imencode(
                "x.png",
                cv2.cvtColor(image.cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            if retval:
                img_dire = Image.open(BytesIO(buffer)).convert("RGB")
                transformed_batch.append(self.transform(img_dire))

        dire = torch.stack(transformed_batch).to(device)
        logits = model(dire)["logits"]
        return Prediction(scores=binary_scores_from_logits(logits), labels=y, logits=logits)


class DNFPredictor(BinaryLogitPredictor):
    def setup(self, model, device):
        from networks.DNF.diffusion import Model

        self.seq = list(map(int, np.linspace(0, 1000, 20 + 1)))
        self.diffusion = Model()
        state_dict = load_weights("networks/weights/dnf_ddim_model-2388000.ckpt", map_location=device)
        self.diffusion.load_state_dict(state_dict)
        self.diffusion = self.diffusion.to(device)
        self.diffusion.eval()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def predict(self, model, batch, device):
        from networks.DNF.utils import inversion_first

        x, y = batch
        dnf = inversion_first(x.float(), self.seq, self.diffusion)

        transformed_batch = []
        for image in dnf:
            ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            in_memory_file = BytesIO()
            im.save(in_memory_file, format="PNG")
            in_memory_file.seek(0)
            transformed_batch.append(self.transform(Image.open(in_memory_file).convert("RGB")))

        dnf = torch.stack(transformed_batch).to(device)
        logits = model(dnf)["logits"]
        return Prediction(scores=binary_scores_from_logits(logits), labels=y, logits=logits)


PREDICTOR_ALIASES = {
    None: "binary",
    "binary": "binary",
    "plain": "binary",
    "utils.validate_plain": "binary",
    "multicls": "softmax",
    "softmax": "softmax",
    "utils.validate_multicls": "softmax",
    "poundnet": "poundnet",
    "utils.validate_poundnet": "poundnet",
    "lnp": "lnp",
    "utils.validate_lnp": "lnp",
    "lgrad": "lgrad",
    "utils.validate_lgrad": "lgrad",
    "dire": "dire",
    "utils.validate_dire": "dire",
    "dnf": "dnf",
    "utils.validate_dnf": "dnf",
}

PREDICTOR_REGISTRY = {
    "binary": BinaryLogitPredictor,
    "softmax": SoftmaxPredictor,
    "poundnet": PoundNetPredictor,
    "lnp": LNPPredictor,
    "lgrad": LGradPredictor,
    "dire": DIREPredictor,
    "dnf": DNFPredictor,
}


def build_predictor(predictor=None, model=None):
    if predictor is None and model is not None and hasattr(model, "forward_binary"):
        predictor = "poundnet"
    key = PREDICTOR_ALIASES.get(predictor, predictor)
    if key not in PREDICTOR_REGISTRY:
        available = ", ".join(sorted(name for name in PREDICTOR_REGISTRY))
        raise KeyError(f'Unknown evaluator predictor "{predictor}". Available predictors: {available}')
    return PREDICTOR_REGISTRY[key]()


def _append_prediction(storage, prediction):
    storage["y_true"].append(prediction.labels.detach().flatten().cpu())
    storage["y_pred"].append(prediction.scores.detach().flatten().cpu())
    if prediction.logits is not None:
        storage["y_logits"].append(prediction.logits.detach().cpu())


def _finalize_predictions(storage):
    y_true = torch.cat(storage["y_true"], dim=0).numpy()
    y_pred = torch.cat(storage["y_pred"], dim=0).numpy()
    if storage["y_logits"]:
        y_logits = torch.cat(storage["y_logits"], dim=0).numpy()
    else:
        y_logits = np.array([])
    metrics = compute_binary_metrics(y_true, y_pred)
    metrics.update({"y_true": y_true, "y_pred": y_pred, "y_logits": y_logits})
    return metrics


def evaluate_model(model, loader, predictor=None, device=None, amp_dtype=None, desc=None, max_batches=None):
    device = device or next(model.parameters()).device
    predictor = build_predictor(predictor, model=model)
    predictor.setup(model, device)

    was_training = model.training
    model.eval()
    storage = {"y_true": [], "y_pred": [], "y_logits": []}
    iterator = tqdm(loader, desc=desc, dynamic_ncols=True)
    grad_context = torch.enable_grad() if predictor.requires_grad else torch.no_grad()

    try:
        with grad_context:
            for batch_idx, batch in enumerate(iterator, start=1):
                if max_batches is not None and batch_idx > max_batches:
                    break
                batch = move_to_device(batch, device)
                with autocast_context(device, amp_dtype):
                    prediction = predictor.predict(model, batch, device)
                _append_prediction(storage, prediction)
    finally:
        model.train(was_training)

    return _finalize_predictions(storage)


def to_train_metrics(result):
    return {
        "val_acc_epoch": result["acc0"],
        "val_ap_epoch": result["ap"],
        "val_racc_epoch": result["r_acc0"],
        "val_facc_epoch": result["f_acc0"],
        "val_auc_epoch": result["auc"],
        "val_f1_epoch": result["f1"],
    }
