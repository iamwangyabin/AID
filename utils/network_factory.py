import torch
import torchvision

from utils.registry import MODELS
def get_model(conf):
    print("Model loaded..")
    # Try registry-based instantiation first
    if hasattr(conf, 'arch') and conf.arch in MODELS:
        kwargs = getattr(conf, 'model_kwargs', {})
        return MODELS.build(conf.arch, **kwargs)
    if conf.arch == 'clip':
        from networks.ojha_detector import CLIPModel
        model = CLIPModel('ViT-L/14')
    elif conf.arch == 'cnn':
        from networks.resnet import resnet50
        model = resnet50(num_classes=1)
    elif conf.arch == 'npr':
        from networks.npr_detector import NPRModel
        model = NPRModel(num_classes=1)
    elif conf.arch == 'freqnet':
        from networks.freqnet_detector import freqnet
        model = freqnet(num_classes=1)
    elif conf.arch == 'FreDect':
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048, 1)
    elif conf.arch == 'Fusing':
        from networks.Fusing.detector import Patch5Model
        model = Patch5Model()
    elif conf.arch == 'Gram':
        from networks.gramnet_detector import resnet18
        model = resnet18(pretrained=True, num_classes=1)
    elif conf.arch == 'poundnet':
        from networks.poundnet_detector import PoundNet
        model = PoundNet(conf)
    elif conf.arch == 'vlp':
        from networks.SPrompts.independentVL import IndepVLPCLIP
        model = IndepVLPCLIP(conf)
    elif conf.arch == 'clipbased':
        from networks.ClipBased.detector import CLIPBasedModel
        model = CLIPBasedModel(pretrained_path=conf.resume)
    elif conf.arch == 'rine':
        from networks.rine_detector import RINEModel
        model = RINEModel(
            backbone = (conf.model.backbone0, conf.model.backbone1),
            nproj = conf.model.nproj,
            proj_dim = conf.model.proj_dim)
    elif conf.arch == 'cascade':
        from networks.cascade_detector import CascadeModel
        model = CascadeModel(num_classes=1)
    elif conf.arch == 'textalign':
        from networks.cascade_detector import CascadeModel
        model = CascadeModel(num_classes=1)
    elif conf.arch == 'sideclip':
        from networks.sideclip_detector import SideCLIP
        model = SideCLIP(conf)
    elif conf.arch == 'aide':
        from networks.AIDE.AIDE import AIDE
        model = AIDE(None, None)

    else:
        from networks.timm_detector import TIMMModel
        model = TIMMModel(conf.arch)
    return model







