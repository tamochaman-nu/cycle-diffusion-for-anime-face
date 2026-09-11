"""
Face parsing (semantic segmentation into skin/hair/background/clothing/etc.)
-- "A-2" of the structure-preserving-translation plan.

Root motivation (from the user's own diagnosis of the FFHQ->anime pipeline):
  problem (2) "背景色・服の色を髪の毛の色と混同してしまう" -- background/clothing
  color gets confused with hair color. Every guidance mechanism tried so far
  (ILVR's pixel low-pass, FBSDiff's DCT frequency-band substitution) is
  spatially/frequency-blind: it has no notion of "this pixel is hair, that
  pixel is background", so color information can leak across a semantic
  boundary the mechanism never sees. A face-parsing mask gives an explicit,
  literal answer to "which pixels are hair vs. background vs. skin vs.
  clothing", which diagnostics/step_e12_translate_matched_arch.py's
  region_color_correct() uses to constrain per-region color statistics
  directly -- see that function's docstring for why this is applied as a
  single POST-PROCESS on the finished image rather than injected into the
  decode loop: every mechanism this session that repeatedly intervened
  inside the loop for hundreds of steps (ILVR, FBSDiff) produced compounding
  artifacts that took significant debugging to resolve (see
  diagnostics/ilvr_failure_mode.md and diagnostics/fbsdiff.py's threshold
  scale note) -- a single corrective pass sidesteps that entire class of bug.

Model: BiSeNet (ResNet18 backbone) trained on CelebAMask-HQ (19 classes),
vendored here (architecture only -- NOT the pretrained weights, which are
downloaded on first use) from https://github.com/yakhyo/face-parsing
(MIT License, Copyright (c) 2024 Yakhyokhuja Valikhujaev). Only the
resnet18 backbone path is vendored (resnet34 omitted, unused).
"""
import os
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

_WEIGHTS_URL = "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.pt"
_WEIGHTS_CACHE_PATH = os.path.expanduser("~/.cache/face_parsing/resnet18.pt")
_INPUT_SIZE = (512, 512)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# CelebAMask-HQ 19-class convention (class 0 = background, 1-18 as listed --
# matches https://github.com/yakhyo/face-parsing/blob/main/utils/common.py
# ATTRIBUTES, verified against this project's fetched copy of that file).
NUM_CLASSES = 19
CLASS_NAMES = [
    "background", "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
    "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip", "neck",
    "neck_l", "cloth", "hair", "hat",
]

# Coarse region groups used by region_color_correct -- chosen to separate
# exactly the categories the user's problem (2) confuses: hair vs.
# background vs. clothing, plus skin as a fourth anchor.
REGION_GROUPS = {
    "background": [0],
    "skin": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "hair": [17],
    "hat": [18],
    "clothing": [14, 15, 16],
}


# --- Vendored ResNet18 backbone (architecture only; weights=None here --
# the full BiSeNet checkpoint loaded in _get_model overwrites every weight,
# so there is no reason to also pay for a separate ImageNet-pretrained
# download of just the backbone). Adapted from
# https://github.com/yakhyo/face-parsing/blob/main/models/resnet.py ---
class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class _ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [_BasicBlock(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes
        for _ in range(1, blocks):
            layers.append(_BasicBlock(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


# --- Vendored BiSeNet (adapted from
# https://github.com/yakhyo/face-parsing/blob/main/models/bisenet.py) ---
class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class _BiSeNetOutput(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, mid_channels, 3, 1)
        self.conv = nn.Conv2d(mid_channels, num_classes, 1, bias=False)

    def forward(self, x):
        return self.conv(self.conv_block(x))


class _AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, out_channels, 3, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.conv_block(x)
        pool = F.avg_pool2d(feat, feat.shape[2:])
        return torch.mul(feat, self.attention(pool))


class _ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNet18Backbone()
        self.arm16 = _AttentionRefinementModule(256, 128)
        self.arm32 = _AttentionRefinementModule(512, 128)
        self.conv_head32 = _ConvBNReLU(128, 128, 3, 1)
        self.conv_head16 = _ConvBNReLU(128, 128, 3, 1)
        self.conv_avg = _ConvBNReLU(512, 128, 1, 1)

    def forward(self, x):
        feat8, feat16, feat32 = self.backbone(x)
        h8, w8 = feat8.shape[2:]
        h16, w16 = feat16.shape[2:]

        avg = F.avg_pool2d(feat32, feat32.shape[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, feat32.shape[2:], mode="nearest")

        feat32_sum = self.arm32(feat32) + avg_up
        feat32_up = F.interpolate(feat32_sum, (h16, w16), mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_sum = self.arm16(feat16) + feat32_up
        feat16_up = F.interpolate(feat16_sum, (h8, w8), mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class _FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = _ConvBNReLU(in_channels, out_channels, 1, 1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.conv_block(torch.cat([fsp, fcp], dim=1))
        attn = F.avg_pool2d(feat, feat.shape[2:])
        attn = self.sigmoid(self.conv2(self.relu(self.conv1(attn))))
        return feat + torch.mul(feat, attn)


class _BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fpn = _ContextPath()
        self.ffm = _FeatureFusionModule(256, 256)
        self.conv_out = _BiSeNetOutput(256, 256, num_classes)
        self.conv_out16 = _BiSeNetOutput(128, 64, num_classes)
        self.conv_out32 = _BiSeNetOutput(128, 64, num_classes)

    def forward(self, x):
        h, w = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.fpn(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)
        feat_out = F.interpolate(self.conv_out(feat_fuse), (h, w), mode="bilinear", align_corners=True)
        return feat_out


_model = None
_model_device = None


def _get_model(device):
    global _model, _model_device
    if _model is None or _model_device != device:
        os.makedirs(os.path.dirname(_WEIGHTS_CACHE_PATH), exist_ok=True)
        if not os.path.exists(_WEIGHTS_CACHE_PATH):
            urllib.request.urlretrieve(_WEIGHTS_URL, _WEIGHTS_CACHE_PATH)
        model = _BiSeNet(NUM_CLASSES)
        state_dict = torch.load(_WEIGHTS_CACHE_PATH, map_location="cpu")
        # The checkpoint carries the original ResNet's unused classification
        # head (fpn.backbone.fc.*) -- our vendored backbone omits it (dead
        # weights never used in forward), so ignore just those extra keys.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        assert not missing, f"unexpected missing keys loading BiSeNet checkpoint: {missing}"
        assert all(k.startswith("fpn.backbone.fc.") for k in unexpected), (
            f"unexpected extra keys loading BiSeNet checkpoint: {unexpected}"
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _model = model.to(device)
        _model_device = device
    return _model


@torch.no_grad()
def parse_class_map(img01, device):
    """img01: (3, H, W) tensor in [0, 1], any resolution. Returns an (H, W)
    long tensor of class indices (0-18, see CLASS_NAMES), at the SAME (H, W)
    as the input (resized internally to BiSeNet's native 512x512 for
    inference, then resized back down via NEAREST to preserve discrete
    class boundaries -- matching the original repo's own convention)."""
    model = _get_model(device)
    h, w = img01.shape[-2], img01.shape[-1]
    x = img01.unsqueeze(0).to(device)
    x = F.interpolate(x, size=_INPUT_SIZE, mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    logits = model(x)
    class_map = logits.argmax(dim=1).float()
    class_map = F.interpolate(class_map.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).squeeze(0)
    return class_map.long()


def region_masks(img01, device):
    """img01: (3, H, W) tensor in [0, 1]. Returns {region_name: (H, W) bool
    tensor} for every name in REGION_GROUPS, at img01's own resolution.
    A region with zero pixels (e.g. no hat/glasses present) still appears
    in the dict with an all-False mask -- callers should skip empty masks."""
    class_map = parse_class_map(img01, device)
    masks = {}
    for name, class_ids in REGION_GROUPS.items():
        mask = torch.zeros_like(class_map, dtype=torch.bool)
        for cid in class_ids:
            mask |= (class_map == cid)
        masks[name] = mask
    return masks
