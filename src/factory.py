"""Model factory: the proposed XLite-UNet plus MONAI baselines retrained on the same split.

DMFNet and nnU-Net are trained with their official code (see README); their predictions
are scored with the same evaluate.py so every number is comparable.
"""
import inspect

from monai.networks import nets

from .model import XLiteUNet


def build_model(cfg):
    name = cfg.get("model", "xlite")
    patch = tuple(cfg.get("patch", [128, 128, 128]))
    if name == "xlite":
        return XLiteUNet(
            in_channels=4, out_channels=3,
            width=cfg.get("width", 16), kernel=cfg.get("kernel", 5), blocks=cfg.get("blocks", 2),
            dropout=cfg.get("dropout", 0.1), deep_supervision=cfg.get("deep_supervision", True),
        )
    if name == "unet3d":  # plain 3D U-Net
        return nets.BasicUNet(spatial_dims=3, in_channels=4, out_channels=3,
                              features=(32, 32, 64, 128, 256, 32))
    if name == "segresnet":  # BraTS 2018 winner backbone (Myronenko)
        return nets.SegResNet(spatial_dims=3, in_channels=4, out_channels=3, init_filters=16,
                              blocks_down=(1, 2, 2, 4), blocks_up=(1, 1, 1), dropout_prob=0.2)
    if name == "unetr":
        return nets.UNETR(in_channels=4, out_channels=3, img_size=patch, feature_size=16,
                          hidden_size=768, mlp_dim=3072, num_heads=12)
    if name == "swinunetr":
        kw = dict(in_channels=4, out_channels=3, feature_size=48, use_checkpoint=True)
        if "img_size" in inspect.signature(nets.SwinUNETR.__init__).parameters:
            kw["img_size"] = patch  # older MONAI versions require it
        return nets.SwinUNETR(**kw)
    raise ValueError(f"unknown model {name}")
