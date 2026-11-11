"""ComfyUI node definitions for the native MangaNinjia port.

Final graph (everything except the 3 nodes below is a stock ComfyUI node):

  CheckpointLoaderSimple --CLIP--> CLIPTextEncode("") ------> MangaNinjiaApply
                        \--VAE---> VAEEncode(reference) ---->/
                        \--VAE------------------------------------> VAEDecode
  LoadImage(lineart) --------------------------------------------> MangaNinjiaApply
  MangaNinjiaLoader ----------------------------------------------> MangaNinjiaApply
  MangaNinjiaPoints x2 -------------------------------------------> MangaNinjiaApply
  MangaNinjiaApply --MODEL/COND/COND--> KSampler(cfg=1.0) --> VAEDecode --> PreviewImage

Requirements: torch only. No diffusers / transformers / cv2 / einops / PIL.
"""
from __future__ import annotations

import os

import folder_paths
import torch
from pathlib import PureWindowsPath
from . import core
from .runtime import (
    WEIGHT_FILES,
    MangaNinjiaBundle,
    RuntimeState,
    find_weight_dirs,
    load_bundle,
    make_calc_cond_batch,
)
import comfy.model_management
# models/MangaNinjia holds the 4 official .pth weights
_MN_DIR = os.path.join(folder_paths.models_dir, "MangaNinjia")
os.makedirs(_MN_DIR, exist_ok=True)
folder_paths.add_model_folder_path("manganinjia", _MN_DIR)
node_mn_cr_path = os.path.dirname(os.path.abspath(__file__))

def _weight_choices():
    dirs = find_weight_dirs()
    if not dirs:
        return (["<no weights found>"], {})
    return ([label for label, _ in dirs], {label: path for label, path in dirs})


class MangaNinjiaLoader:
    """Load the 4 diffusers-named MangaNinjia weights into native comfy objects."""

    DESCRIPTION = (
        "Loads denoising_unet.pth / reference_unet.pth / controlnet.pth / point_net.pth "
        "from models/MangaNinjia using ComfyUI's own UNet + ControlNet loaders. "
        "The SD1.5 checkpoint is NOT needed here (its CLIP/VAE come from the stock "
        "CheckpointLoaderSimple); MangaNinjia ships its own finetuned UNet."
    )
    CATEGORY = "MangaNinjia"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_dir": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("MANGANINJIA",)
    RETURN_NAMES = ("manganinjia",)
    FUNCTION = "load"

    def load(self, model_dir):
        # An explicit path wins; empty means "the default models/MangaNinjia".
        path = None
        model_dir = PureWindowsPath(model_dir).as_posix().strip() if model_dir else ""
        if model_dir:
            if os.path.isdir(model_dir) and all(
                os.path.isfile(os.path.join(model_dir, f)) for f in WEIGHT_FILES.values()
            ):
                path = os.path.abspath(model_dir)
            else:
                raise RuntimeError(
                    "model_dir does not contain all 4 MangaNinjia weights "
                    "(denoising_unet.pth / reference_unet.pth / controlnet.pth / "
                    "point_net.pth): {}".format(model_dir)
                )
        if path is None:
            _, mapping = _weight_choices()
            path = mapping.get(".") or next(iter(mapping.values()), None)
        if path is None:
            raise RuntimeError(
                "MangaNinjia weights not found. Put denoising_unet.pth, "
                "reference_unet.pth, controlnet.pth, point_net.pth into "
                "ComfyUI/models/MangaNinjia/ or set model_dir to that folder."
            )
        return (load_bundle(path),)


class MangaNinjiaApply:
    """Patch a MODEL so the stock KSampler performs MangaNinjia's 3-branch CFG."""

    DESCRIPTION = (
        "Produces a MODEL that can be sampled with the stock KSampler. It replaces "
        "calc_cond_batch with one batch=3 UNet forward (uncond / reference / point), "
        "runs the reference UNet once at timestep 0 for mutual self-attention, and "
        "applies the native lineart ControlNet hint in the upstream [-1,1] range. "
        "Set KSampler cfg to 1.0 (any value is numerically harmless). "
        "empty_prompt is OPTIONAL: when unconnected the node loads the bundled "
        "sd15_empty_saved_cond.pt (a pre-baked CLIPTextEncode('') output), so no CLIP "
        "text encoder has to be loaded at all. "
        "clip_vision is OPTIONAL too: when connected it must take the OUTPUT of a "
        "native CLIPVisionEncode whose IMAGE input is the COLOR REFERENCE image (the "
        "same LoadImage that goes to VAEEncode -> reference_latent), NOT the lineart, "
        "with crop=center. CLIPVisionLoader must be CLIP-ViT-L/14 (224px/patch14) and "
        "its file must contain visual_projection.weight so image_embeds is 768-d; "
        "ViT-H/bigG (1280-d) is rejected. With clip_vision unconnected the empty-prompt "
        "token is used for all 3 branches, so colour guidance comes only from the "
        "reference latent and the point pairs. When the reference and the "
        "lineart show different views, sparse point correspondences are essential "
        "for colour transfer: the official demo uses 14 matched pairs, 50 steps and "
        "seed 0. This model is very sensitive to the initial noise: changing only "
        "the seed from 0 to 44 turns pink hair / red clothes into greenish skin. "
        "Load manganinjia_workflow_demo.json to reproduce the official result. "
        "No points (or only 2-3) produces arbitrary / greenish colours."
    )
    CATEGORY = "MangaNinjia"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manganinjia": ("MANGANINJIA",),
                "reference_latent": ("LATENT",),
                "lineart": ("IMAGE",),
                "guidance_scale_ref": ("FLOAT", {"default": 9.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "guidance_scale_point": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "lineart_threshold": ("FLOAT", {"default": core.LINEART_THRESHOLD, "min": 0.0, "max": 1.0, "step": 0.01}),
                "point_size": ("INT", {"default": core.DEFAULT_POINT_SIZE, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION_OUTPUT",),
                "empty_prompt": ("CONDITIONING",),
                "points_ref": ("MINJIA_POINTS",),
                "points_target": ("MINJIA_POINTS",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("model", "positive", "negative")
    FUNCTION = "apply"

    def apply(self, manganinjia, reference_latent, lineart,
              guidance_scale_ref, guidance_scale_point, lineart_threshold,
              point_size, clip_vision=None, empty_prompt=None,
              points_ref=None, points_target=None):
        bundle: MangaNinjiaBundle = manganinjia
        target_device = comfy.model_management.get_torch_device()
        # uncond branch: first token of the empty-prompt CLIP-L hidden states
        # (upstream prompt2embeds takes text_encoder(ids)[0][:, 0, :]).
        # sd15_empty_saved_cond.pt is a pre-baked CLIPTextEncode("") output, so
        # no CLIP text encoder (and no checkpoint CLIP) has to be loaded.
        if empty_prompt is None:
            empty_prompt = torch.load(
                os.path.join(node_mn_cr_path, "sd15_empty_saved_cond.pt"),
                map_location="cpu",
            )
        uncond = empty_prompt[0][0][:, 0:1, :].float().to(device=target_device)
        if uncond.shape[-1] != 768:
            raise RuntimeError(
                "MangaNinjia (SD1.5) needs a 768-d CLIP-L conditioning, got {}. "
                "Connect CLIPTextEncode from the SD1.5 checkpoint's CLIP.".format(uncond.shape[-1])
            )
        if clip_vision is None:
            # No vision encoder: use the empty-prompt token for every branch.
            # Colour guidance then comes only from the reference latent + points.
            image_embeds = uncond
        else:
            emb = clip_vision["image_embeds"]
            if emb.ndim == 1:
                emb = emb[None]
            # IMPORTANT: comfy's CLIPVision.encode_image always parks
            # image_embeds on intermediate_device() (CPU unless --gpu-only),
            # not on the sampling device. torch.cat in build_context would then
            # mix cpu + cuda -> "Expected all tensors to be on the same device".
            image_embeds = emb[:, None, :].float().to(device=target_device)
            if image_embeds.shape[-1] != 768:
                raise RuntimeError(
                    "MangaNinjia needs CLIP-ViT-L/14 image_embeds (768-d), got {}. "
                    "Leave clip_vision unconnected to use the empty-prompt fallback, "
                    "or load a clip-vit-large-patch14 vision model.".format(image_embeds.shape[-1])
                )
        context = core.build_context(uncond, image_embeds)

        # lineart hint: [0,1] IMAGE -> [-1,1] BCHW-like BHWC, batch=3 for the CFG branches
        h, w = int(lineart.shape[1]), int(lineart.shape[2])
        hint = core.lineart_hint(lineart, lineart_threshold, batch=3)
        cnet = bundle.controlnet.copy()
        cnet.set_cond_hint(hint, strength=1.0)

        # point towers: 512-canvas clicks scaled to the real lineart size
        p_ref = core.build_point_image(core.parse_points(points_ref), h, w, source_size=point_size)
        p_main = core.build_point_image(core.parse_points(points_target), h, w, source_size=point_size)
        with torch.no_grad():
            point_ref = [t.detach() for t in bundle.point_net(p_ref)]
            point_main = [t.detach() for t in bundle.point_net(p_main)]

        state = RuntimeState()
        state.ref_latent = reference_latent["samples"]
        state.context = context
        state.point_ref = point_ref
        state.point_main = point_main
        state.controlnet = cnet

        calc = make_calc_cond_batch(
            bundle, state, float(guidance_scale_ref), float(guidance_scale_point)
        )
        model = bundle.main_model.clone()
        model.set_model_sampler_calc_cond_batch_function(calc)
        return (model, empty_prompt, empty_prompt)


class MangaNinjiaPoints:
    """Sparse correspondence points, edited by the modern DOM-widget frontend."""

    DESCRIPTION = (
        "Click matching points on the reference / lineart canvases in the node UI. "
        "Points are stored as JSON in the points_json widget -- no custom HTTP route, "
        "no per-filename node cache. The same order on both nodes defines pairs."
    )
    CATEGORY = "MangaNinjia"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "points_json": ("STRING", {"default": "[]", "multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("MINJIA_POINTS",)
    RETURN_NAMES = ("points",)
    FUNCTION = "parse"

    def parse(self, points_json, image=None):
        return (core.parse_points(points_json),)


NODE_CLASS_MAPPINGS = {
    "MangaNinjiaLoader": MangaNinjiaLoader,
    "MangaNinjiaApply": MangaNinjiaApply,
    "MangaNinjiaPoints": MangaNinjiaPoints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MangaNinjiaLoader": "MangaNinjia Loader (native)",
    "MangaNinjiaApply": "MangaNinjia Apply (native)",
    "MangaNinjiaPoints": "MangaNinjia Points",
}
