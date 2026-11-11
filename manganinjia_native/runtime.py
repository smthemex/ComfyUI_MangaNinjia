"""ComfyUI-native runtime: weight loading + three-branch denoise predictor.

All model construction goes through stock ComfyUI loaders:
  comfy.sd.load_diffusion_model_state_dict   (diffusers UNet -> native UNet)
  comfy.controlnet.load_controlnet_state_dict (diffusers ControlNet -> native ControlNet)
  comfy.utils.load_torch_file                 (safetensors / .pth)
PointNet is a plain nn.Module. No diffusers / transformers objects survive.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch

from . import core, patches

WEIGHT_FILES = {
    "denoising": "denoising_unet.pth",
    "reference": "reference_unet.pth",
    "controlnet": "controlnet.pth",
    "point": "point_net.pth",
}


class MangaNinjiaBundle:
    def __init__(self, main_model, ref_model, controlnet, point_net, weights_dir):
        self.main_model = main_model          # comfy ModelPatcher (denoising UNet)
        self.ref_model = ref_model            # comfy ModelPatcher (reference UNet, t=0 only)
        self.controlnet = controlnet          # comfy ControlNet (native ControlBase)
        self.point_net = point_net            # nn.Module
        self.weights_dir = weights_dir

    def get_extra_models(self):
        out = [self.ref_model]
        out += self.controlnet.get_models()
        return out


def find_weight_dirs():
    """List every directory under models/MangaNinjia that has the 4 weights."""
    import folder_paths

    out = []
    seen = set()
    for root in folder_paths.get_folder_paths("manganinjia"):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if all(f in filenames for f in WEIGHT_FILES.values()):
                rel = os.path.relpath(dirpath, root)
                label = "." if rel == "." else rel.replace("\\", "/")
                if label not in seen:
                    seen.add(label)
                    out.append((label, os.path.abspath(dirpath)))
    return out


def load_bundle(weights_dir: str) -> MangaNinjiaBundle:
    import comfy.controlnet
    import comfy.sd
    import comfy.utils

    paths = {k: os.path.join(weights_dir, v) for k, v in WEIGHT_FILES.items()}
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise FileNotFoundError("MangaNinjia weight missing: {}".format(p))

    # comfy's loader pops keys out of the dict, so hand it a shallow copy each time.
    denoising_sd = comfy.utils.load_torch_file(paths["denoising"])
    reference_sd = comfy.utils.load_torch_file(paths["reference"])
    controlnet_sd = comfy.utils.load_torch_file(paths["controlnet"])
    point_sd = comfy.utils.load_torch_file(paths["point"])

    main_model = comfy.sd.load_diffusion_model_state_dict(dict(denoising_sd))
    if main_model is None:
        raise RuntimeError("Could not detect SD1.5 UNet in denoising_unet.pth")

    # RefUNet has no conv_norm_out / conv_out in its checkpoint (upstream
    # comments them out). Their outputs are never used -- only the attention
    # bank is -- but fill them from the denoising UNet to avoid random buffers.
    for k in ("conv_norm_out.weight", "conv_norm_out.bias", "conv_out.weight", "conv_out.bias"):
        if k not in reference_sd and k in denoising_sd:
            reference_sd[k] = denoising_sd[k]
    ref_model = comfy.sd.load_diffusion_model_state_dict(dict(reference_sd))
    if ref_model is None:
        raise RuntimeError("Could not detect SD1.5 UNet in reference_unet.pth")

    controlnet = comfy.controlnet.load_controlnet_state_dict(dict(controlnet_sd))
    if controlnet is None:
        raise RuntimeError("Could not load MangaNinjia controlnet.pth")

    point_net = core.PointNet()
    missing, unexpected = point_net.load_state_dict(point_sd, strict=True)
    point_net.eval().requires_grad_(False)

    logging.info(
        "[MangaNinjia] native load ok: %d/%d/%d/%d tensors, weights=%s",
        len(denoising_sd), len(reference_sd), len(controlnet_sd), len(point_sd), weights_dir,
    )
    return MangaNinjiaBundle(main_model, ref_model, controlnet, point_net, weights_dir)


class RuntimeState:
    """Mutable state shared by the patch closures (never stored in model_options).

    ComfyUI deep-copies model_options on every clone; plain function objects are
    returned as-is by copy.deepcopy, so closures keep pointing at this object.
    That is why all tensors live here instead of in model_options.
    """

    def __init__(self):
        self.bank = {}
        self.mode = {}
        self.q_len = {}
        self.point_ref: List[torch.Tensor] = []
        self.point_main: List[torch.Tensor] = []
        self.ref_latent: Optional[torch.Tensor] = None
        self.context: Optional[torch.Tensor] = None
        self.last_sigma = None
        self.ref_ready = False
        self.read_patch = None
        self.replace_patch = None
        self.controlnet = None
        self.stats = {"read": 0, "replace": 0, "modes": {}}
        self.stats_logged = False

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


def make_calc_cond_batch(bundle: MangaNinjiaBundle, state: RuntimeState,
                         guidance_ref: float, guidance_point: float):
    """Return a `sampler_calc_cond_batch_function` so stock KSampler just works.

    It fully replaces calc_cond_batch: one batch=3 UNet forward per step, then
    the upstream three-branch guidance combination. Returning [den, den] makes
    cfg_function an identity for ANY cfg value (both entries are equal).
    """
    def calc(args):
        import comfy.model_management as mm

        model = args["model"]                 # BaseModel (already loaded by KSampler)
        x = args["input"]                    # [1,4,h,w] scaled latent
        sigma = args["sigma"]                 # [1]
        device = x.device
        dtype = x.dtype
        s = float(sigma.reshape(-1)[0].item())

        with torch.no_grad():
            # A rising sigma (or a cold state) means a fresh sampling run:
            # re-run the reference UNet once at timestep 0, exactly as upstream.
            if (not state.ref_ready) or state.last_sigma is None or s > state.last_sigma + 1e-6:
                _refresh_reference(bundle, state, x, sigma, device, dtype,
                                   getattr(model, "current_patcher", None))
            state.last_sigma = s

            ctx = state.context.to(device=device, dtype=dtype)
            x3 = x.repeat(3, 1, 1, 1)
            t3 = sigma.reshape(1).repeat(3).to(device=device, dtype=dtype)

            # Native comfy control channel. cond is a plain dict so
            # ControlNet.get_control can read c_crossattn as a tensor.
            ctrl = state.controlnet.get_control(
                x3, t3, {"c_crossattn": ctx}, 3, {}
            )

            read_opts = patches.build_options(
                state, write=False, sigma=t3,
                read_patch=state.read_patch, replace_patch=state.replace_patch,
            )
            den3 = model.apply_model(
                x3, t3, c_crossattn=ctx, control=ctrl,
                transformer_options=read_opts,
            )
            den = core.combine_denoised(den3, guidance_ref, guidance_point)
            if s < 0.05 and not state.stats_logged:
                logging.info("[MangaNinjia] patch call stats: %s", state.stats)
                state.stats_logged = True
            return [den, den]

    return calc


def _refresh_reference(bundle, state, x, sigma, device, dtype, current_patcher=None):
    """Run reference_unet once with sigma=0 and cache all self-attention banks."""
    import comfy.model_management as mm

    # Ask comfy's memory manager for all three models at once. On this machine
    # comfy uses dynamic paged models which refuse to evict each other, so the
    # explicit call is needed or the reference UNet stays on CPU / gets evicted.
    models = [bundle.ref_model] + state.controlnet.get_models()
    if current_patcher is not None:
        models.append(current_patcher)
    try:
        mm.load_models_gpu(models)
    except Exception as e:  # pragma: no cover
        logging.warning("[MangaNinjia] load_models_gpu warning: %s", e)

    state.bank.clear()
    state.mode.clear()
    state.q_len.clear()

    # ComfyUI's VAEEncode returns an *unscaled* latent (std ~5.2 for SD1.5),
    # while KSampler feeds UNets a scaled one (`process_latent_in`, x0.18215).
    # The official pipeline scales with rgb_latent_scale_factor before running
    # reference_unet, so the same scaling must happen here -- otherwise the
    # reference UNet sees a ~5.5x too-large latent and every bank is garbage.
    ref_latent = state.ref_latent.to(device=device, dtype=dtype)
    ref_latent = bundle.main_model.model.process_latent_in(ref_latent)
    ref_latent = core.resize_latent(ref_latent, x.shape[-2], x.shape[-1])
    ref_x = ref_latent.repeat(3, 1, 1, 1)
    t0 = torch.zeros(3, device=device, dtype=dtype)
    write_opts = patches.build_options(state, write=True, sigma=t0)
    ctx = state.context.to(device=device, dtype=dtype)
    bundle.ref_model.model.apply_model(
        ref_x, t0, c_crossattn=ctx, transformer_options=write_opts
    )

    state.controlnet.model_sampling_current = bundle.main_model.model.model_sampling
    state.controlnet.timestep_range = None

    if state.read_patch is None:
        state.read_patch = patches.make_read_patch(state)
        state.replace_patch = patches.make_replace_patch(state)
    state.ref_ready = True
    logging.info("[MangaNinjia] reference bank captured at %d layers", len(state.bank))
