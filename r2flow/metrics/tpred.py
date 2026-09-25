"""A domain-trained feature extractor for LiDAR range images.

The paper uses off-the-shelf SemanticKITTI/ShapeNet extractors, which is right for
evaluating a KITTI-360 generator. Ours generates indoor quadruped scans with a
0.34 m median depth, far outside that regime, so we fit our own on the same domain
as the generative target -- the same relationship the paper has, different data.

The pretext task is predicting the flow timestep `t` from `x_t`, which needs no
labels and reuses the training forward process. On its own that task is close to
trivial: the local pixel variance of `x_t` is about (1-t)^2, so a shallow network
solves it with a single feature direction, and a rank-1 representation is useless
as a distribution metric. The auxiliary denoising head is what forces the encoder
to keep scene content, and the 32-bin head (rather than a lone scalar regression)
is what stops the representation collapsing onto the noise-level axis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..models.ops import Pad


def _ringify(module: nn.Module) -> nn.Module:
    """Replace padded convs/pools with circular-azimuth, replicate-elevation padding.

    Azimuth wraps, so a seam at column 0 is an artefact. Elevation does not: row 0
    is the horizon and the last row is nadir, and wrapping nadir into the horizon
    would be geometrically wrong -- hence replicate vertically.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.Conv2d, nn.MaxPool2d)):
            padding = child.padding
            if isinstance(padding, str) or padding == 0 or padding == (0, 0):
                continue
            ph, pw = (padding, padding) if isinstance(padding, int) else padding
            if ph == 0 and pw == 0:
                continue
            child.padding = (0, 0) if isinstance(child, nn.Conv2d) else 0
            # Pad takes (left, right, top, bottom).
            setattr(
                module,
                name,
                nn.Sequential(Pad((pw, pw, ph, ph), ring=True, mode="replicate"), child),
            )
        else:
            _ringify(child)
    return module


class _DenoiseDecoder(nn.Module):
    """Bottleneck vector -> a full-resolution reconstruction. Discarded at eval.

    Reads the pooled BOTTLENECK, not the feature map. Decoding from the map lets
    the network reconstruct while the bottleneck carries nothing but noise level,
    which is exactly the collapse this head exists to prevent -- measured as a
    participation ratio of 4.2 (healthy is 15-40) when it was wired to the map.
    Forcing reconstruction through 128 numbers makes scene content the only way to
    satisfy it.
    """

    START = (8, 16)

    def __init__(self, bottleneck_dim: int, out_channels: int):
        super().__init__()
        in_channels = 128
        self.in_channels = in_channels
        self.project = nn.Linear(
            bottleneck_dim, in_channels * self.START[0] * self.START[1]
        )
        widths = [in_channels, 256, 128, 64, 32, 16]
        # H x8 and W x32, so elevation stops doubling once it reaches full height.
        factors = [(2, 2), (2, 2), (2, 2), (1, 2), (1, 2)]
        blocks = []
        for i, factor in enumerate(factors):
            blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=factor, mode="nearest"),
                    Pad((1, 1, 1, 1), ring=True, mode="replicate"),
                    nn.Conv2d(widths[i], widths[i + 1], 3),
                    nn.GroupNorm(8, widths[i + 1]),
                    nn.GELU(),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(widths[-1], out_channels, 1)

    def forward(self, bottleneck):
        x = self.project(bottleneck).view(-1, self.in_channels, *self.START)
        return self.head(self.blocks(x))


class TimestepPredictor(nn.Module):
    """timm backbone + timestep heads + a denoising head.

    `features()` returns the 128-d bottleneck, averaged over four azimuth rolls so
    the descriptor is exactly invariant to robot heading.
    """

    NUM_BINS = 32

    def __init__(
        self,
        backbone: str = "resnet10t",
        in_channels: int = 1,
        pretrained: bool = True,
        feature_dim: int = 128,
        num_rolls: int = 4,
    ):
        super().__init__()
        import timm

        self.backbone_name = backbone
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.num_rolls = num_rolls

        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="",
        )
        # Stock ResNets downsample 32x in both axes, which would crush 64 elevation
        # rows to 2. Halve the vertical stride twice in the stem so elevation only
        # drops 8x; azimuth has 8x more columns and absorbs the rest.
        stem = model.conv1
        stem_convs = [m for m in stem.modules() if isinstance(m, nn.Conv2d)]
        if not stem_convs:
            raise ValueError(f"no stem conv found in {backbone}")
        stem_convs[0].stride = (1, 2)
        if hasattr(model, "maxpool") and isinstance(model.maxpool, nn.MaxPool2d):
            model.maxpool.stride = (1, 2)
        self.backbone = _ringify(model)
        self.num_channels = model.num_features

        self.neck = nn.Sequential(
            nn.LayerNorm(self.num_channels),
            nn.Linear(self.num_channels, feature_dim),
            nn.GELU(),
        )
        self.bin_head = nn.Linear(feature_dim, self.NUM_BINS)
        self.reg_head = nn.Linear(feature_dim, 1)
        self.decoder = _DenoiseDecoder(feature_dim, in_channels)

    def forward_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def _bottleneck(self, feature_map: torch.Tensor) -> torch.Tensor:
        pooled = feature_map.mean(dim=(2, 3))
        return self.neck(pooled)

    def forward(self, x: torch.Tensor) -> dict:
        """Training forward: all heads."""
        feature_map = self.forward_map(x)
        bottleneck = self._bottleneck(feature_map)
        return {
            "features": bottleneck,
            "bins": self.bin_head(bottleneck),
            "t": self.reg_head(bottleneck).squeeze(-1).sigmoid(),
            "denoised": self.decoder(bottleneck),
        }

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation descriptor: the bottleneck, averaged over azimuth rolls."""
        width = x.shape[-1]
        shifts = [round(i * width / self.num_rolls) for i in range(self.num_rolls)]
        stacked = [self._bottleneck(self.forward_map(x.roll(s, dims=-1))) for s in shifts]
        return torch.stack(stacked).mean(0)

    @torch.no_grad()
    def predict_t(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["t"]


def soft_bin_targets(t: torch.Tensor, num_bins: int, sigma_bins: float = 0.5):
    """Gaussian-smoothed bin targets.

    A hard one-hot target injects noise for any `t` near a bin boundary, which at
    32 bins is most of them.
    """
    centres = (torch.arange(num_bins, device=t.device) + 0.5) / num_bins
    logits = -0.5 * ((t[:, None] - centres[None, :]) / (sigma_bins / num_bins)) ** 2
    return torch.softmax(logits, dim=-1)


def save(model: TimestepPredictor, path, extra: dict | None = None):
    torch.save(
        {
            "cfg": {
                "backbone": model.backbone_name,
                "in_channels": model.in_channels,
                "feature_dim": model.feature_dim,
                "num_rolls": model.num_rolls,
                **(extra or {}),
            },
            "weights": model.state_dict(),
        },
        path,
    )


def load(path, device="cpu") -> tuple[TimestepPredictor, dict]:
    """Mirrors r2flow.utils.inference.setup_model's checkpoint convention."""
    ckpt = torch.load(path, map_location="cpu")
    cfg = dict(ckpt["cfg"])
    model = TimestepPredictor(
        backbone=cfg["backbone"],
        in_channels=cfg["in_channels"],
        pretrained=False,
        feature_dim=cfg["feature_dim"],
        num_rolls=cfg.get("num_rolls", 4),
    )
    model.load_state_dict(ckpt["weights"])
    model.eval().to(device)
    return model, cfg
