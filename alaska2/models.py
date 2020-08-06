import sys
from pathlib import Path

import pycls
import timm
import torch
from efficientnet_pytorch import EfficientNet
from pycls.core.builders import build_model as pycls_build_model
from pycls.core.config import cfg, load_cfg


def get_backbone(backbone: str, num_classes: int = 4):
    out_channels = {
        "efficientnet-b0": 1280,
        "efficientnet-b2": 1408,
        "efficientnet-b3": 1536,
        "efficientnet-b5": 2048,
        "efficientnet-b7": 2560,
        "regnetx-1.6gf": 912,
        "pyconvhgresnet50": 2048,
        "pyconvhgresnet101": 2048,
        "pyconvhgresnet152": 2048,
        "pyconvresnet50": 2048,
        "pyconvresnet101": 2048,
        "pyconvresnet152": 2048,
    }

    if backbone.startswith("efficientnet"):
        return EfficientNet.from_pretrained(backbone), out_channels[backbone]

    if backbone.startswith("regnet"):
        pretrained_urls = {
            "regnetx-1.6gf": "https://dl.fbaipublicfiles.com/pycls/dds_baselines/160990626/RegNetX-1.6GF_dds_8gpu.pyth"
        }
        config_dir = Path(pycls.__file__).parent.parent / "configs/dds_baselines/" / backbone[:7]

        if backbone == "regnetx-1.6gf":
            load_cfg(config_dir, "RegNetX-1.6GF_dds_8gpu.yaml")
        else:
            raise ValueError
        cfg.freeze()

        model = pycls_build_model()
        model.load_state_dict(torch.utils.model_zoo.load_url(pretrained_urls[backbone], map_location="cpu"))

        return model, out_channels[backbone]

    if backbone.startswith("timm-"):
        return timm.create_model(backbone[5:], pretrained=True, num_classes=num_classes), None

    if backbone.startswith("pyconvhgresnet"):
        if "pyconv" not in sys.path:
            sys.path.append("pyconv")
        from pyconv.models import pyconvhgresnet

        model = getattr(pyconvhgresnet, backbone)(pretrained=True)
        del model.avgpool
        del model.fc
        return model, out_channels[backbone]

    if backbone.startswith("pyconvresnet"):
        if "pyconv" not in sys.path:
            sys.path.append("pyconv")
        from pyconv.models import pyconvresnet

        model = getattr(pyconvresnet, backbone)(pretrained=True)
        del model.avgpool
        del model.fc
        return model, out_channels[backbone]

    raise ValueError


def get_feature_of(name: str, model, x):
    if name.startswith("efficientnet"):
        return model.extract_features(x)
    elif name.startswith("regnet"):
        modules = list(model.children())[:-1]  # skip head
        for module in modules:
            x = module(x)
        return x
    elif name.startswith("pyconv"):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

    raise ValueError


# based on https://github.com/qubvel/segmentation_models.pytorch
def patch_first_conv(model, in_channels, reuse: bool = False):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    old_weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = old_weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = old_weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(module.out_channels, module.in_channels // module.groups, *module.kernel_size)

    module.weight = torch.nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()

    if in_channels > 3 and reuse:
        weight = module.weight
        weight[:, :3] = old_weight
        module.weight = torch.nn.parameter.Parameter(weight)


def patch_first_conv_stride(model):
    # decrease first conv's stride
    modules_iter = iter(model.modules())
    for module in modules_iter:
        if isinstance(module, torch.nn.Conv2d) and tuple(module.stride) == (2, 2):
            break

    module.stride = (1, 1)

    for module in modules_iter:
        if isinstance(module, torch.nn.Conv2d) and tuple(module.stride) == (2, 2):
            break

    module.stride = (3, 3)


class Model(torch.nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone = backbone
        self.model, out_channels = get_backbone(backbone)

        if not self.backbone.startswith("timm-"):
            self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.linear = torch.nn.Linear(out_channels, num_classes)

    def get_feature(self, x):
        if self.backbone.startswith("timm-"):
            raise NotImplementedError
        return get_feature_of(self.backbone, self.model, x)

    def forward(self, x):
        if self.backbone.startswith("timm-"):
            return self.model(x)
        else:
            x = self.get_feature(x)
            x = self.avg_pool(x).reshape(x.shape[0], -1)
            return self.linear(x)
