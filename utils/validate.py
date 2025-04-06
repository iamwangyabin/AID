import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score

from copy import deepcopy
from tqdm import tqdm

import torch
import torch.utils.data
from torchvision import transforms
from torch.nn import functional as F

def validate(y_true, y_pred):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"
    N = y_true.shape[0]

    if y_pred[0:N // 2].max() <= y_pred[N // 2:N].min():  # perfectly separable case
        return (y_pred[0:N // 2].max() + y_pred[N // 2:N].min()) / 2

    best_acc = 0
    best_thres = 0
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp >= thres] = 1
        temp[temp < thres] = 0

        acc = (temp == y_true).sum() / N
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc

    return best_thres


def calculate_acc_auc_f1(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0
    try:
        f1 = f1_score(y_true, y_pred>thres)
    except:
        f1 = 0
    ap = average_precision_score(y_true, y_pred)
    return r_acc, f_acc, acc, auc, f1, ap



def validate_plain(model, loader):
    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()
            logits = model(in_tens)['logits']
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict



def validate_multicls(model, loader):
    # 这就是个两个分类头的binary 分类器该如何实现val
    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()
            logits = model(in_tens)['logits']
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(F.softmax(logits, 1)[:,1].flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict





def validate_poundnet(model, loader, specific_cls=False):

    if specific_cls:
        from networks.poundnet_detector import load_clip_to_cpu
        clip_model = load_clip_to_cpu(model.cfg)
        token_embedding = clip_model.token_embedding


    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label in tqdm(loader):
            in_tens = img.cuda()
            if specific_cls:
                logits = model.forward_binary_classnames(in_tens, specific_cls, token_embedding)['logits']
            else:
                logits = model.forward_binary(in_tens)['logits']
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(F.softmax(logits, 1)[:,1].flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict

def validate_lnp(model_dis, data_loader):
    from networks.LNP.denoising_rgb import DenoiseNet
    from collections import OrderedDict

    model_restoration = DenoiseNet()
    checkpoint = torch.load('networks/weights/sidd_rgb.pth')
    try:
        model_restoration.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model_restoration.load_state_dict(new_state_dict)

    model_restoration.cuda()
    model_restoration.eval()

    y_true, y_pred, y_logits = [], [], []
    i = 0
    for data in data_loader:
        i += 1
        print("batch number {}/{}".format(i, len(data_loader)), end='\r')
        input_img = data[0].cuda().to(torch.float32)
        label = data[1].cuda()

        rgb_restored = model_restoration(input_img)
        rgb_restored = torch.round(torch.clamp(rgb_restored, 0, 1)*255.)*255.
        rgb_restored = torch.clamp(rgb_restored, 0, 255) / 255.0

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        rgb_restored = (rgb_restored - mean) / std

        with torch.no_grad():
            logits = model_dis(rgb_restored)['logits']

        y_logits.extend(logits.flatten().tolist())
        y_pred.extend(logits.sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())

    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict

def validate_lgrad(model_dis, data_loader):
    from networks.LGrad import build_model

    gen_model = build_model(gan_type='stylegan', module='discriminator', resolution=256, label_size=0, image_channels=3)
    gen_model.load_state_dict(torch.load('networks/weights/karras2019stylegan-bedrooms-256x256_discriminator.pth'), strict=True)
    gen_model.cuda()
    gen_model.eval()


    y_true, y_pred, y_logits = [], [], []
    i = 0
    for data in data_loader:
        i += 1
        print("batch number {}/{}".format(i, len(data_loader)), end='\r')
        input_img = data[0].cuda().to(torch.float32)  # [batch_size, 3, height, width]
        label = data[1].cuda()  # [batch_size, 1]
        input_img.requires_grad = True
        pre = gen_model(input_img)
        gen_model.zero_grad()
        grads = torch.autograd.grad(pre.sum(), input_img, create_graph=True, retain_graph=True, allow_unused=False)[0]

        b_min = torch.min(grads.view(grads.size(0), -1), dim=1)[0]
        grads = grads - b_min.view(-1, 1, 1, 1)
        b_max = torch.max(grads.view(grads.size(0), -1), dim=1)[0]
        grads = grads/b_max.view(-1, 1, 1, 1)
        grads = grads*255.

        grads = F.interpolate(grads, size=(224, 224), mode='bilinear', align_corners=False)

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

        mean = torch.tensor(MEAN).view(1, 3, 1, 1).cuda() # Reshape to [1, 3, 1, 1] for broadcasting
        std = torch.tensor(STD).view(1, 3, 1, 1).cuda() # Reshape to [1, 3, 1, 1] for broadcasting
        grads = torch.clamp(grads, 0, 255) / 255.0
        normalized_grads = (grads - mean) / std
        with torch.no_grad():
            logits = model_dis(normalized_grads)['logits']
        y_logits.extend(logits.flatten().tolist())
        y_pred.extend(logits.sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())

    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict

def validate_dire(model_dis, data_loader):
    from networks.DIRE.script_util import create_model_and_diffusion

    defaults = dict(
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        dropout=0.1,
        image_size=256,
        learn_sigma=True,
        noise_schedule='linear',
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
        timestep_respacing='ddim20',
        channel_mult='',
        num_heads=4,
        num_heads_upsample=-1,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_new_attention_order=False
    )
    diffusion_model, diffusion = create_model_and_diffusion(**defaults)

    diffusion_model.load_state_dict(torch.load('weights/256x256_diffusion_uncond.pt', map_location="cpu"))
    diffusion_model.cuda()
    diffusion_model.convert_to_fp16()
    diffusion_model.eval()
    reverse_fn = diffusion.ddim_reverse_sample_loop


    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        i = 0
        for data in data_loader:
            i += 1
            print("batch number {}/{}".format(i, len(data_loader)), end='\r')
            input_img = data[0].cuda().to(torch.float32)
            label = data[1].cuda()

            latent = reverse_fn(diffusion_model,input_img.shape,noise=input_img,clip_denoised=True,real_step=0,)
            recons = diffusion.ddim_sample_loop(diffusion_model,latent.shape,noise=latent,clip_denoised=True,real_step=0,)
            dire = torch.abs(input_img - recons)

            dire = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8)
            dire = dire.permute(0, 2, 3, 1)
            dire = dire.contiguous()

            # Define the transformation
            trans = transforms.Compose([
                # transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            transformed_batch = []
            for ii in range(len(dire)):
                lab = label[ii].item()
                if lab == 0:
                    retval, buffer = cv2.imencode("x.png", cv2.cvtColor(dire[ii].cpu().numpy().astype(np.uint8),
                                                                       cv2.COLOR_RGB2BGR), )#[cv2.IMWRITE_JPEG_QUALITY, 75])
                else:
                    retval, buffer = cv2.imencode("x.png", cv2.cvtColor(dire[ii].cpu().numpy().astype(np.uint8),
                                                                       cv2.COLOR_RGB2BGR), )#[cv2.IMWRITE_JPEG_QUALITY, 100])

                if retval:
                    img_dire = Image.open(BytesIO(buffer)).convert('RGB')
                    dire_transformed = trans(img_dire)
                    transformed_batch.append(dire_transformed)

            dire = torch.stack(transformed_batch).cuda()
            logits = model_dis(dire)['logits']

            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

            # predictions = [1 if p > 0.5 else 0 for p in y_pred]  # Convert sigmoid outputs to binary predictions
            # correct_predictions = sum(p == t for p, t in zip(predictions, y_true))
            # accuracy = correct_predictions / len(y_true)
            # print(f"Accuracy of this batch: {accuracy * 100:.2f}%")

    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict

def validate_dnf(model_dis, data_loader):
    # Something wrong, can't find
    from networks.DNF.diffusion import Model
    from networks.DNF.utils import inversion_first
    from torchvision.utils import make_grid

    seq = list(map(int, np.linspace(0, 1000, 20+1)))

    diffusion = Model()
    diffusion.load_state_dict(torch.load('networks/weights/dnf_ddim_model-2388000.ckpt'))
    diffusion = diffusion.cuda()
    diffusion.eval()

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        y_true, y_pred, y_logits = [], [], []
        i = 0
        for data in data_loader:
            i += 1
            print("batch number {}/{}".format(i, len(data_loader)), end='\r')
            input_img = data[0].cuda().to(torch.float32)
            label = data[1].cuda()
            dnf = inversion_first(input_img, seq, diffusion)

            transformed_batch = []
            for ii in range(len(dnf)):

                grid = make_grid(dnf[ii])
                ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                im = Image.fromarray(ndarr)
                im.save('o.jpg', format='JPEG')
                import pdb;pdb.set_trace()

                lab = label[ii].item()
                if lab == 0:
                    in_memory_file = BytesIO()
                    im.save(in_memory_file, format='JPEG')
                else:
                    in_memory_file = BytesIO()
                    im.save(in_memory_file, format='PNG')
                in_memory_file.seek(0)
                img_dnf = Image.open(in_memory_file).convert('RGB')
                dnf_transformed = trans(img_dnf)
                transformed_batch.append(dnf_transformed)

            dnf = torch.stack(transformed_batch).cuda()
            logits = model_dis(dnf)['logits']
            y_logits.extend(logits.flatten().tolist())
            y_pred.extend(logits.sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred, y_logits = np.array(y_true), np.array(y_pred), np.array(y_logits)
    r_acc0, f_acc0, acc0, auc, f1, ap = calculate_acc_auc_f1(y_true, y_pred, 0.5)
    num_real = (y_true == 0).sum()
    num_fake = (y_true == 1).sum()
    result_dict = { 'ap': ap, 'auc': auc, 'f1': f1, 'r_acc0': r_acc0, 'f_acc0': f_acc0, 'acc0': acc0,
        'num_real': num_real, 'num_fake': num_fake, 'y_true': y_true, 'y_pred': y_pred, 'y_logits': y_logits }
    return result_dict
