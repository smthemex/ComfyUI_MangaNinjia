"""Reference-attention (mutual self attention) as native ComfyUI UNet patches.

Why this fits the stock UNet exactly
------------------------------------
Upstream replaces BasicTransformerBlock.forward and calls `attn1` as a
cross-attention with three inputs:

    attn1(hidden_states,                       # query source
          encoder_hidden_states=...,          # key source
          encoder_hidden_states_v=...)        # value source

ComfyUI's `attn1_patch` hook has exactly the same three slots before the
to_q / to_k / to_v projections:

    n, context_attn1, value_attn1 = patch(n, context_attn1, value_attn1, opts)
    # (without patches_replace) CrossAttention.forward does
    # q = to_q(x); k = to_k(context); v = to_v(value)

So the projections stay 100% native.  The only thing we take over is the
attention math itself, via `patches_replace["attn1"]`, because the three CFG
branches need different key/value lengths (L vs 2L) and cannot share one SDPA
call.  We call comfy's own optimized_attention three times and concatenate.

Branch semantics copied verbatim from hacked_basic_transformer_inner_forward:
  0 uncond : pure self attention                 q=n0, k=v=n0
  1 ref    : mutual attention                   q=n1, k=v=cat(n1, bank1)
  2 point  : point-guided mutual attention     q=n2+pm,
                                               k=cat(n2+pm, bank2+pr),
                                               v=cat(n2,    bank2)
If no point tower matches a layer's (tokens, channels), branches 1 and 2 both
degrade to plain mutual attention -- this reproduces the upstream `except` path.
"""
from __future__ import annotations

import torch

from .core import filter_by_size

try:
    from comfy.ldm.modules import attention as _comfy_attention
except Exception:  # pragma: no cover - allows offline import for parity scripts
    _comfy_attention = None


def _attn(q, k, v, heads, extra_options):
    """Use comfy's selected attention backend (xformers / pytorch / pytorch-sdpa)."""
    # Slices such as k[0:1, :L] are non-contiguous views. Some GPU backends
    # (xformers / flash) silently mis-handle them; make them explicit.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    precision = extra_options.get("attn_precision", None)
    if _comfy_attention is not None:
        fn = _comfy_attention.optimized_attention
        try:
            return fn(q, k, v, heads, attn_precision=precision)
        except TypeError:
            return fn(q, k, v, heads)
    # Offline fallback: plain PyTorch SDPA with comfy's [B, L, (H*D)] layout.
    B, L, D = q.shape
    dh = D // heads

    def reshape(t):
        return t.view(B, -1, heads, dh).transpose(1, 2)

    return torch.nn.functional.scaled_dot_product_attention(
        reshape(q), reshape(k), reshape(v)
    ).transpose(1, 2).reshape(B, -1, D)


def bank_key(extra_options):
    block = extra_options.get("block")
    if block is None:
        return None
    return (tuple(block), int(extra_options.get("block_index", 0)))


def replace_key(extra_options):
    block = extra_options.get("block")
    return tuple(block) if block is not None else None


def make_write_patch(state):
    """reference_unet side: cache post-LayerNorm self-attention inputs."""

    def attn1_patch(n, context_attn1, value_attn1, extra_options):
        k = bank_key(extra_options)
        if k is not None:
            state["bank"][k] = n.detach().clone()
        return n, context_attn1, value_attn1

    return attn1_patch


def make_read_patch(state):
    """denoising_unet side: build the three-branch q/k/v *inputs*."""

    def attn1_patch(n, context_attn1, value_attn1, extra_options):
        stats = getattr(state, "stats", None)
        if stats is not None:
            stats["read"] = stats.get("read", 0) + 1
        k = bank_key(extra_options)
        if k is None:
            state["mode"][k] = "identity"
            return n, context_attn1, value_attn1

        bank = state["bank"].get(k)
        if bank is None or n.shape[0] != 3:
            state["mode"][k] = "identity"
            return n, context_attn1, value_attn1

        state["q_len"][k] = n.shape[1]
        pr = filter_by_size(state["point_ref"], bank)
        pm = filter_by_size(state["point_main"], bank)

        if pr and pm:
            state["mode"][k] = "point"
            pref = pr[0].to(device=n.device, dtype=n.dtype)
            pmain = pm[0].to(device=n.device, dtype=n.dtype)
            bank = bank.to(device=n.device, dtype=n.dtype)
            n0, n1, n2 = n[0:1], n[1:2], n[2:3]
            b0, b1, b2 = bank[0:1], bank[1:2], bank[2:3]
            q_in = torch.cat([n0, n1, n2 + pmain], dim=0)
            k_in = torch.cat([
                torch.cat([n0, b0], dim=1),
                torch.cat([n1, b1], dim=1),
                torch.cat([n2 + pmain, b2 + pref], dim=1),
            ], dim=0)
            v_in = torch.cat([
                torch.cat([n0, b0], dim=1),
                torch.cat([n1, b1], dim=1),
                torch.cat([n2, b2], dim=1),
            ], dim=0)
            return q_in, k_in, v_in

        state["mode"][k] = "mutual"
        bank = bank.to(device=n.device, dtype=n.dtype)
        q_in = n
        k_in = torch.cat([n, bank], dim=1)
        v_in = torch.cat([n, bank], dim=1)
        return q_in, k_in, v_in

    return attn1_patch


def make_replace_patch(state):
    """denoising_unet side: three SDPA calls, one per guidance branch."""

    def attn1_replace(q, k, v, extra_options):
        stats = getattr(state, "stats", None)
        key = bank_key(extra_options)
        mode = state["mode"].get(key, "identity")
        if stats is not None:
            stats["replace"] = stats.get("replace", 0) + 1
            stats["modes"][mode] = stats["modes"].get(mode, 0) + 1
        heads = int(extra_options.get("n_heads", 8))
        if mode == "identity":
            return _attn(q, k, v, heads, extra_options)

        L = state["q_len"].get(key, q.shape[1])
        out0 = _attn(q[0:1], k[0:1, :L], v[0:1, :L], heads, extra_options)
        out1 = _attn(q[1:2], k[1:2], v[1:2], heads, extra_options)
        out2 = _attn(q[2:3], k[2:3], v[2:3], heads, extra_options)
        return torch.cat([out0, out1, out2], dim=0)

    return attn1_replace


def all_replace_keys():
    """Register one patch per possible SD1.5 transformer location.

    ComfyUI looks up (section, id, block_index) first and then falls back to
    the (section, id) tuple, so these tuples cover every attn1 layer. Extra
    keys that never occur are harmless.
    """
    keys = [("middle", 0)]
    for i in range(16):
        keys.append(("input", i))
        keys.append(("output", i))
    return keys


def build_options(state, *, write: bool, sigma=None, read_patch=None, replace_patch=None):
    """Build a fresh transformer_options dict for one UNet call."""
    opts = {
        "cond_or_uncond": [0, 0, 0],
        "sigmas": sigma,
    }
    if write:
        opts["patches"] = {"attn1_patch": [make_write_patch(state)]}
    else:
        opts["patches"] = {"attn1_patch": [read_patch]} if read_patch else {}
        opts["patches_replace"] = {
            "attn1": {k: replace_patch for k in all_replace_keys()}
        }
    return opts
