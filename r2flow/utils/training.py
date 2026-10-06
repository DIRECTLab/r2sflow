import math
import re
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from rich.console import Console
from rich.table import Table
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def pseudo_huber_loss(v_t, u_t, coef=0.00054):
    """
    The coefficient 0.00054 is designed for image domains in [2]
    [1] Improving the Training of Rectified Flows (https://arxiv.org/abs/2405.20320)
    [2] Improved Techniques for Training Consistency Models (https://arxiv.org/abs/2310.14189)
    """
    assert v_t.shape == u_t.shape
    _, C, H, W = v_t.shape
    const = coef * math.sqrt(C * H * W)
    loss = torch.sqrt((v_t - u_t).pow(2) + const**2) - const
    loss = loss.mean()
    return loss


# class SampledCoupling(torch.utils.data.Dataset):
#     def __init__(self, sample_dir):
#         self.sample_dir = Path(sample_dir)
#         self.files = list(sorted(self.sample_dir.glob("*.pth")))

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         path = self.files[idx]
#         x_1 = torch.load(path, map_location="cpu")
#         x_0_seed = int(re.search(r"\d{10}", str(path)).group())
#         return x_0_seed, x_1

#     def __repr__(self):
#         return f"#data: {len(self.files):,}"


class IndexedSamples(torch.utils.data.Dataset):
    def __init__(self, sample_dir):
        self.sample_dir = Path(sample_dir)
        self.files = list(sorted(self.sample_dir.glob("*.pth")))
        print(f"Found {len(self.files):,} samples!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        x_1_dict = torch.load(path, map_location="cpu")
        x_0_seed = int(re.search(r"\d{10}", str(path)).group())
        return x_0_seed, x_1_dict

    def __repr__(self):
        return f"#data: {len(self.files):,}"


def restore_x_0(seeds, shape, device):
    zs = []
    for seed in seeds.cpu().tolist():
        rng = torch.Generator(device=device).manual_seed(seed)
        zs.append(torch.randn(*shape, device=device, generator=rng))
    return torch.stack(zs, dim=0)


def _detect_diff(cfg_1: Any, cfg_2: Any, path: str = "") -> Dict[str, tuple]:
    if type(cfg_1) is not type(cfg_2):
        return {path or "<root>": (cfg_1, cfg_2)}
    if not is_dataclass(cfg_1):
        return {path or "<root>": (cfg_1, cfg_2)} if cfg_1 != cfg_2 else {}

    diffs: Dict[str, tuple] = {}
    for f in fields(cfg_1):
        av, bv = getattr(cfg_1, f.name), getattr(cfg_2, f.name)
        child_path = f"{path}.{f.name}" if path else f.name
        if is_dataclass(av):
            diffs.update(_detect_diff(av, bv, child_path))
        elif av != bv:
            diffs[child_path] = (av, bv)
    return diffs


# Same release as hubconf._get_url
_RELEASE_URL = "https://github.com/kazuto1011/r2flow/releases/download/weights/{}.pth"


def load_pretrained_weights(
    model: torch.nn.Module, source: str, channel_index: List[int], cfg: Any = None
):
    """Load R2Flow weights into a freshly built HDiT, adapting the channel count.

    `source` is a checkpoint path or a release name (e.g. "r2flow-kitti360-1rf").
    Only the tokenizer conv and the detokenizer linear depend on the number of
    image channels; they keep the checkpoint channels in `channel_index` (e.g.
    [0] to keep depth from a depth+reflectance model). The `coords` buffer is
    skipped since the caller sets the target sensor's ray angles. Any other
    mismatch raises instead of silently loading a partial model.
    """
    from .option import DefaultConfig

    if Path(source).exists():
        ckpt = torch.load(source, map_location="cpu")
    else:
        ckpt = torch.hub.load_state_dict_from_url(
            _RELEASE_URL.format(source), map_location="cpu"
        )
    if cfg is not None:
        show_diff(DefaultConfig(**ckpt["cfg"]), cfg)

    src = dict(ckpt["weights"])
    src.pop("coords", None)
    dst = model.state_dict()
    if set(src) != set(dst) - {"coords"}:
        raise ValueError(
            f"key mismatch: missing {sorted(set(dst) - set(src) - {'coords'})}, "
            f"unexpected {sorted(set(src) - set(dst))}"
        )

    num_patch = model.patch_size[0] * model.patch_size[1]
    sliced = []
    for key, value in src.items():
        if value.shape == dst[key].shape:
            continue
        if key == "tokenizer.0.weight":  # (D, C, P1, P2)
            value = value[:, channel_index]
        elif key == "detokenizer.1.weight":  # (P1 * P2 * C, D), laid out (P1 P2 C)
            value = value.view(num_patch, -1, value.shape[-1])[:, channel_index]
            value = value.reshape(-1, value.shape[-1])
        if value.shape != dst[key].shape:
            raise ValueError(
                f"{key}: checkpoint {tuple(src[key].shape)} vs model {tuple(dst[key].shape)}"
            )
        src[key] = value
        sliced.append(key)

    model.load_state_dict(src, strict=False)  # strict apart from the skipped coords
    print(f"Loaded {len(src):,} tensors from {source} (channel-sliced: {sliced or 'none'})")


def show_diff(last_cfg: Any, current_cfg: Any):
    diff = _detect_diff(last_cfg, current_cfg)
    if not diff:
        Console().print("No differences found.")
        return
    table = Table(title="Configuration", show_lines=True)
    table.add_column("Item", style="cyan", no_wrap=True)
    table.add_column("Last", style="magenta")
    table.add_column("Current", style="green")
    for path, (last, current) in diff.items():
        table.add_row(path, repr(last), repr(current))
    Console().print(table)
