"""XLite-UNet: a slim 3D U-shaped network for BraTS region segmentation.

Design (see the project plan):
  stem -> Encoder1 (C) -> Encoder2 (2C) -> Encoder3 (4C) -> Bottleneck (8C, MC-dropout)
       -> Decoder3 -> Decoder2 -> Decoder1 -> 1x1 head: 3 sigmoid maps [WT, TC, ET]
Each stage uses DS blocks: depthwise k^3 conv -> GroupNorm -> pointwise expand (x2) -> GELU
-> pointwise project, with a residual connection. Skips are added (not concatenated) to
keep the decoder light. Deep-supervision heads on Decoder2/Decoder3 are used in training only.
"""
import torch
import torch.nn as nn


def _gn(c):
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c)


class DSBlock(nn.Module):
    def __init__(self, c, k=5, expand=2):
        super().__init__()
        self.dw = nn.Conv3d(c, c, k, padding=k // 2, groups=c)
        self.norm = _gn(c)
        self.pw1 = nn.Conv3d(c, c * expand, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv3d(c * expand, c, 1)

    def forward(self, x):
        return x + self.pw2(self.act(self.pw1(self.norm(self.dw(x)))))


def _stage(c, n, k):
    return nn.Sequential(*[DSBlock(c, k) for _ in range(n)])


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 2, stride=2)
        self.norm = _gn(cout)

    def forward(self, x):
        return self.norm(self.conv(x))


class Up(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.ConvTranspose3d(cin, cout, 2, stride=2)
        self.norm = _gn(cout)

    def forward(self, x, skip):
        return self.norm(self.conv(x)) + skip


class XLiteUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, width=16, kernel=5, blocks=2,
                 dropout=0.1, deep_supervision=True):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.deep_supervision = deep_supervision
        self.stem = nn.Sequential(nn.Conv3d(in_channels, c1, 3, padding=1), _gn(c1), nn.GELU())
        self.enc1 = _stage(c1, blocks, kernel)
        self.down1 = Down(c1, c2)
        self.enc2 = _stage(c2, blocks, kernel)
        self.down2 = Down(c2, c3)
        self.enc3 = _stage(c3, blocks, kernel)
        self.down3 = Down(c3, c4)
        self.bottleneck = _stage(c4, blocks, kernel)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        self.up3 = Up(c4, c3)
        self.dec3 = _stage(c3, 1, kernel)
        self.up2 = Up(c3, c2)
        self.dec2 = _stage(c2, 1, kernel)
        self.up1 = Up(c2, c1)
        self.dec1 = _stage(c1, 1, kernel)
        self.head = nn.Conv3d(c1, out_channels, 1)
        if deep_supervision:
            self.aux2 = nn.Conv3d(c2, out_channels, 1)
            self.aux3 = nn.Conv3d(c3, out_channels, 1)

    def forward(self, x):
        s1 = self.enc1(self.stem(x))
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        b = self.drop(self.bottleneck(self.down3(s3)))
        d3 = self.dec3(self.up3(b, s3))
        d2 = self.dec2(self.up2(d3, s2))
        d1 = self.dec1(self.up1(d2, s1))
        out = self.head(d1)
        if self.training and self.deep_supervision:
            return [out, self.aux2(d2), self.aux3(d3)]
        return out

    # layers available as Grad-CAM targets
    def cam_layers(self):
        return {"dec1": self.dec1, "dec2": self.dec2, "dec3": self.dec3, "bottleneck": self.bottleneck}


def enable_mc_dropout(model):
    """Switch only dropout layers to train mode (for Monte-Carlo dropout at inference)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout3d)):
            m.train()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
