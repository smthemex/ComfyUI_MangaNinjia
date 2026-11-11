"""Pure-torch core for the native MangaNinjia port.

No diffusers / transformers / cv2 / PIL / einops / omegaconf here.
Everything in this file can be imported outside ComfyUI for numeric parity checks.

The upstream MangaNinjia pipeline (D:/Downloads/MangaNinjia-main) is a SD1.5
line-art colorizer built out of:
  * denoising_unet  - a finetuned SD1.5 UNet (the model we sample with)
  * reference_unet  - a second SD1.5 UNet run ONCE at timestep 0; its
                      post-LayerNorm self-attention inputs are cached ("bank")
  * controlnet      - a finetuned SD1.5 lineart ControlNet
  * point_net       - 4 conv towers turning sparse correspondence points into
                      per-resolution additive embeddings
  * CLIP-ViT-L/14 image_embeds as the (single token) cross-attention condition
  * a three-branch guidance rule (uncond / reference / point)
"""
from __future__ import annotations

import json
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

POINT_OUT_CHANNELS = (320, 640, 1280, 1280)
POINT_DOWNSAMPLES = (6, 2, 2, 2)
DEFAULT_POINT_SIZE = 512
LINEART_THRESHOLD = 0.24


class PointNet(nn.Module):
    """1:1 structural copy of src/point_network.py (keys are checkpoint keys).

    downsample=6 -> three stride-2 convs (H/8), then 2 -> one conv each.
    For a 512x512 point map this yields (4096,320), (1024,640), (256,1280),
    (64,1280), which is exactly what the four SD1.5 attention resolutions need.
    """

    def __init__(
        self,
        conditioning_channels: int = 1,
        out_channels: Sequence[int] = POINT_OUT_CHANNELS,
        downsamples: Sequence[int] = POINT_DOWNSAMPLES,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        current = conditioning_channels
        for oc, downsample in zip(out_channels, downsamples):
            layers = []
            for _ in range(downsample // 2):
                layers.append(nn.Conv2d(current, oc, kernel_size=3, stride=2, padding=1))
                layers.append(nn.SiLU())
                current = oc
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        embeddings = []
        h = x
        for block in self.blocks:
            h = block(h)
            B, C, H, W = h.shape
            embeddings.append(h.view(B, C, H * W).transpose(1, 2))
        return embeddings


def parse_points(value, default=None) -> List[Tuple[int, int]]:
    """Accept a MINJIA_POINTS list, a JSON string, or None -> []."""
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        value = json.loads(s)
    out = []
    for item in value:
        if isinstance(item, dict):
            x, y = item.get("x"), item.get("y")
        else:
            x, y = item[0], item[1]
        out.append((int(round(float(x))), int(round(float(y)))))
    return out


def build_point_image(
    points: Sequence[Tuple[int, int]],
    height: int,
    width: int,
    *,
    source_size: int = DEFAULT_POINT_SIZE,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Sparse point ids -> [1,1,H,W] float map, matching infer.py's numpy matrix.

    Upstream hardcodes a 512x512 uint8 matrix (index+1 at each click) and feeds
    it (as float) through PointNet. New coordinates are recorded on a 512x512
    canvas in the browser, so they are scaled to the actual image size here.
    """
    m = torch.zeros((1, 1, height, width), dtype=torch.float32)
    sx = width / float(source_size)
    sy = height / float(source_size)
    for idx, (x, y) in enumerate(points):
        xi = int(round(x * sx))
        yi = int(round(y * sy))
        if 0 <= xi < width and 0 <= yi < height:
            m[0, 0, yi, xi] = float((idx % 255) + 1)
    return m.to(device=device, dtype=dtype)


def filter_by_size(embeddings: Sequence[torch.Tensor], reference: torch.Tensor) -> List[torch.Tensor]:
    """Upstream filter_matrices_by_size: match the last two dims (tokens, channels)."""
    shape = tuple(reference.shape[-2:])
    return [e for e in embeddings if tuple(e.shape[-2:]) == shape]


def lineart_hint(image: torch.Tensor, threshold: float = LINEART_THRESHOLD, batch: int = 3) -> torch.Tensor:
    """ComfyUI IMAGE [B,H,W,3] in [0,1] -> ControlNet hint [3,3,H,W] in [-1,1].

    ControlBase.set_cond_hint expects BCHW (stock ControlNetApplyAdvanced calls
    image.movedim(-1, 1) before set_cond_hint); common_upscale also treats the
    last two axes as H/W without any layout conversion. BHWC here would make
    F.interpolate see C=512 and blow up in input_hint_block.

    Upstream (single_infer): PIL convert('L') -> /255 -> values <= 0.24 zeroed
    -> repeat to 3 channels -> *2-1.  The ControlNet is trained on that exact
    [-1,1] range; comfy's usual ControlNetApply feeds [0,1], so we must not use
    the stock node for this hint.
    """
    rgb = image[..., :3].to(torch.float32)
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gray = gray.unsqueeze(-1)
    gray = torch.where(gray <= threshold, torch.zeros_like(gray), gray)
    hint = (gray * 2.0 - 1.0).expand(-1, -1, -1, 3).movedim(-1, 1)  # -> BCHW
    if hint.shape[0] == 1 and batch != 1:
        hint = hint.expand(batch, -1, -1, -1)
    return hint.contiguous()


def build_context(uncond: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
    """[1,1,768] x 2 -> [3,1,768] = [uncond, reference, point]."""
    return torch.cat([uncond, image_embeds, image_embeds], dim=0).contiguous()


def combine_denoised(denoised: torch.Tensor, guidance_ref: float, guidance_point: float) -> torch.Tensor:
    """Upstream three-branch rule, applied in denoised space (affine-equivariant).

    noise_pred_1 = uncond + g_ref * (ref - uncond)
    noise_pred_2 = ref    + g_point * (point - ref)
    noise_pred   = (noise_pred_1 + noise_pred_2) / 2
    denoised = x - sigma * noise_pred, and both branches have coefficient sum 1,
    so the identical linear combination works directly on denoised tensors.
    """
    u, r, p = denoised.chunk(3, dim=0)
    d1 = u + guidance_ref * (r - u)
    d2 = r + guidance_point * (p - r)
    return (d1 + d2) * 0.5


def resize_latent(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Align the reference latent to the sampling latent geometry."""
    if latent.shape[-2:] == (height, width):
        return latent
    return F.interpolate(latent, size=(height, width), mode="bilinear", align_corners=False)
