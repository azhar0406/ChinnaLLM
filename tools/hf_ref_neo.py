#!/usr/bin/env python3
"""
HF Q4-roundtrip reference for GPT-Neo (TinyStories-Instruct).

Loads the HF model via transformers, runs greedy generation for HF_REF,
then roundtrips all 2D weights through Q4 quantization and runs again for Q4_REF.
Prints HF_REF and Q4_REF lines with 30 generated token ids and per-step top1 logit values.

With --pruned, masks logits to the keep-set and remaps token IDs through the
vocab pruning mapping, mimicking the pruned model's output space.

Usage:
    python tools/hf_ref_neo.py [model_dir] [--q4-only] [--pruned <remap.json>]
    model_dir defaults to models/tinystories-instruct-3M
"""
import sys
import os
import argparse
import copy
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from export_q4 import fp16_to_uint16


def fp16_ceil_scale(val):
    # NOTE: returns single float32 — differs from export_q4_neo.py (uint16_bits, float32) tuple variant
    half = np.float16(val)
    if float(half) < float(val):
        half = np.nextafter(half, np.float16(np.inf), dtype=np.float16)
    return float(half.astype(np.float32))

GROUP_SIZE = 64


def fp16_to_f32(h):
    """Decode IEEE binary16 to float32."""
    h = int(h)
    if h == 0:
        return 0.0
    if h == 0x8000:
        return -0.0
    if h == 0x7C00:
        return float('inf')
    if h == 0xFC00:
        return float('-inf')
    if (h & 0x7C00) == 0x7C00:
        if h & 0x03FF:
            return float('nan')
        return float('-inf') if h & 0x8000 else float('inf')
    sign = (h >> 15) & 1
    exponent = (h >> 10) & 0x1F
    mantissa = h & 0x3FF
    if exponent == 0:
        val = mantissa * (2.0 ** -24)
        return -val if sign else val
    else:
        val = (mantissa | 0x400) * (2.0 ** (exponent - 15 - 10))
        return -val if sign else val


def fp16_ceil_to_f32(val):
    half = np.float16(val)
    if float(half) < float(val):
        half = np.nextafter(half, np.float16(np.inf), dtype=np.float16)
    return float(half.astype(np.float32))


def quantize_q4_spec_np(arr, group_size=GROUP_SIZE):
    """Quantize a 2D float32 numpy array to SPEC_Q4 format. Returns packed, scales_fp16."""
    out_dim, in_dim = arr.shape
    assert in_dim % group_size == 0, (
        f"in_dim {in_dim} must be divisible by group_size {group_size}. Shape: {arr.shape}"
    )
    n_groups = in_dim // group_size
    packed = np.zeros((out_dim, in_dim // 2), dtype=np.uint8)
    scales = np.zeros((out_dim, n_groups), dtype=np.uint16)

    for o in range(out_dim):
        for g in range(n_groups):
            gs = g * group_size
            ge = gs + group_size
            group_vals = arr[o, gs:ge]

            amax = float(np.abs(group_vals).max())
            if amax < 1e-12:
                amax = 1.0
            scale = amax / 7.0
            scale_f32 = fp16_ceil_scale(scale)
            scales[o, g] = int(np.float16(scale_f32).view(np.uint16))

            q = np.clip(np.round(group_vals / scale_f32), -7, 7).astype(np.int8)
            for j in range(group_size):
                nibble = (int(q[j]) + 8) & 0xF
                bi = gs // 2 + j // 2
                if j % 2 == 0:
                    packed[o, bi] = nibble & 0xF
                else:
                    packed[o, bi] |= (nibble & 0xF) << 4

    return packed, scales.flatten()


def _dequant_packed(packed, scales, out_ch, in_ch, group_size):
    """Dequantize packed Q4 tensor back to float32."""
    n_groups = in_ch // group_size
    result = np.zeros((out_ch, in_ch), dtype=np.float32)
    packed_2d = packed.reshape(out_ch, in_ch // 2) if packed.ndim == 1 else packed
    for o in range(out_ch):
        for g in range(n_groups):
            scale = float(fp16_to_f32(int(scales[o * n_groups + g])))
            base = g * (group_size // 2)
            for j in range(group_size):
                bi = base + j // 2
                byte_val = packed_2d[o, bi]
                nibble = (byte_val & 0xF) if (j % 2 == 0) else (byte_val >> 4)
                result[o, g * group_size + j] = (int(nibble) - 8) * scale
    return result


def dequant_q4_roundtrip(arr, group_size=GROUP_SIZE):
    """Quantize a 2D float32 array to Q4 then dequantize back to float32."""
    if arr.ndim != 2:
        return arr

    out_dim, in_dim = arr.shape
    pad = 0
    if in_dim % group_size != 0:
        pad = group_size - (in_dim % group_size)
        arr = np.pad(arr, ((0, 0), (0, pad)), mode='constant', constant_values=0)
    packed, scales = quantize_q4_spec_np(arr, group_size)
    result = _dequant_packed(packed, scales, out_dim, arr.shape[1], group_size)
    if pad > 0:
        result = result[:, :in_dim]
    return result


def dequant_q8_roundtrip(arr, group_size=GROUP_SIZE):
    """8-bit symmetric per-group quantization (Q8g64 per SPEC_Q4_NEO v4).
    Groups of group_size consecutive values along in_dim share one fp16 scale.
    Mimics the binary exporter: scale = max|w|/127, round, clip ±127,
    stores the fp16 CEILED scale (matching export_q4_neo.py)."""
    if arr.ndim != 2:
        return arr
    out_dim, in_dim = arr.shape
    pad = 0
    if in_dim % group_size != 0:
        pad = group_size - (in_dim % group_size)
        arr = np.pad(arr, ((0, 0), (0, pad)), mode='constant', constant_values=0)
    n_groups = arr.shape[1] // group_size
    result = np.zeros_like(arr, dtype=np.float32)
    for o in range(out_dim):
        for g in range(n_groups):
            gs = g * group_size
            ge = gs + group_size
            group_vals = arr[o, gs:ge]
            amax = float(np.abs(group_vals).max())
            if amax < 1e-12:
                amax = 1.0
            scale = amax / 127.0
            scale_f32 = fp16_ceil_scale(scale)
            q = np.clip(np.round(group_vals / scale_f32), -127, 127).astype(np.int8)
            result[o, gs:ge] = q.astype(np.float32) * scale_f32
    if pad > 0:
        result = result[:, :in_dim]
    return result


def is_attn_param(name):
    return any(s in name for s in ['attn.attention.q_proj', 'attn.attention.k_proj',
                                     'attn.attention.v_proj', 'attn.attention.out_proj'])


def is_mlp_param(name):
    return any(s in name for s in ['mlp.c_fc', 'mlp.c_proj'])


def q4_roundtrip_model(model):
    """Clone the model with all 2D weight tensors Q4-roundtripped (Q4 g=64)."""
    import torch
    m_q4 = copy.deepcopy(model)
    for name, param in m_q4.named_parameters():
        if param.ndim == 2:
            w_np = param.detach().cpu().to(torch.float32).numpy()
            w_q4_np = dequant_q4_roundtrip(w_np)
            param.data.copy_(torch.from_numpy(w_q4_np))
    return m_q4


def mixed_quant_model(model):
    """Clone the model with winning mixed scheme: mlp+attn=Q8, rest=Q4g64."""
    import torch
    m_mix = copy.deepcopy(model)
    for name, param in m_mix.named_parameters():
        if param.ndim == 2:
            w_np = param.detach().cpu().to(torch.float32).numpy()
            if is_mlp_param(name) or is_attn_param(name):
                w_q_np = dequant_q8_roundtrip(w_np)
            else:
                w_q_np = dequant_q4_roundtrip(w_np)
            param.data.copy_(torch.from_numpy(w_q_np))
    return m_mix


def all_q8_model(model):
    """Clone the model with ALL 2D tensors Q8-roundtripped (Q8g64 everywhere)."""
    import torch
    m_q8 = copy.deepcopy(model)
    for name, param in m_q8.named_parameters():
        if param.ndim == 2:
            w_np = param.detach().cpu().to(torch.float32).numpy()
            w_q_np = dequant_q8_roundtrip(w_np)
            param.data.copy_(torch.from_numpy(w_q_np))
    return m_q8


def load_remap(remap_path):
    """Load the vocab pruning remap. Returns (old_to_new, new_to_old, new_eos)."""
    with open(remap_path) as f:
        remap = json.load(f)
    # old_to_new: dict with string keys -> int values
    old_to_new = {int(k): v for k, v in remap['old_to_new'].items()}
    # new_to_old: list where index = new_id, value = old_id
    new_to_old = remap['new_to_old']
    new_eos = remap['eos_token_id']
    return old_to_new, new_to_old, new_eos


def greedy_generate_transformers(model, prompt_ids, max_new, old_to_new=None):
    """Run greedy generation using the transformers model.

    If old_to_new is provided, masks logits to only allow tokens in the keep-set
    and remaps output IDs through the mapping.
    Returns (gen_ids_old, gen_ids_new, top1_logits) if old_to_new else (gen_ids, top1_logits).
    """
    import torch
    model.eval()
    input_ids = torch.tensor([prompt_ids])

    gen_old = []
    gen_new = []
    top1_logits = []

    with torch.no_grad():
        for step in range(max_new):
            out = model(input_ids)
            logits = out.logits[0, -1, :].clone()

            if old_to_new is not None:
                keep_mask = torch.full_like(logits, float('-inf'))
                for old_id in old_to_new.keys():
                    keep_mask[old_id] = 0.0
                logits = logits + keep_mask

            next_token = int(torch.argmax(logits).item())
            top1_logit = float(logits[next_token].item())
            gen_old.append(next_token)
            if old_to_new is not None:
                gen_new.append(old_to_new.get(next_token, -1))
            top1_logits.append(top1_logit)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

    if old_to_new is not None:
        return gen_old, gen_new, top1_logits
    return gen_old, top1_logits


PROMPTS = [
    'Summary: a story about a brave cat\nWords: cat, jump, tree\nStory:',
    'Summary: a story about a happy dog\nWords: dog, park, ball\nStory:',
    'Summary: a story about two friends\nWords: friends, share, kindness\nStory:',
]


def main():
    parser = argparse.ArgumentParser(description='HF and Q4-roundtrip reference for GPT-Neo')
    parser.add_argument('model_dir', nargs='?', default='models/tinystories-instruct-3M',
                        help='Path to model directory (default: models/tinystories-instruct-3M)')
    parser.add_argument('--q4-only', action='store_true',
                        help='Skip HF reference, only run Q4 roundtrip')
    parser.add_argument('--pruned', default=None,
                        help='Path to vocab remap JSON for pruned-space output')
    parser.add_argument('--scheme', default='mixed', choices=['all-q4g64', 'mixed', 'all-q8'],
                        help='Quantization scheme for Q4_REF (default: mixed = mlp+attn Q8, rest Q4g64)')
    args = parser.parse_args()

    model_dir = args.model_dir

    import torch
    from transformers import GPTNeoForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = GPTNeoForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    max_new = 30

    old_to_new = None
    new_to_old = None
    if args.pruned:
        old_to_new, new_to_old, _ = load_remap(args.pruned)
        K = len(new_to_old)
        print(f"=== Pruned mode: K={K} keep tokens from {args.pruned} ===")

    if not args.q4_only:
        print("=== HF Reference (fp32, transformers greedy) ===")
        for i, prompt in enumerate(PROMPTS):
            prompt_ids = tok.encode(prompt)
            result = greedy_generate_transformers(model, prompt_ids, max_new, old_to_new)

            if old_to_new is not None:
                gen_old, gen_new, logits = result
                logit_strs = [f"{v:.4f}" for v in logits]
                print(f"HF_REF_{i}_old: {gen_old}")
                print(f"HF_REF_{i}_new: {gen_new}")
                print(f"HF_LOGITS_{i}: {' '.join(logit_strs)}")
            else:
                gen_ids, logits = result
                logit_strs = [f"{v:.4f}" for v in logits]
                print(f"HF_REF_{i}: {gen_ids}")
                print(f"HF_LOGITS_{i}: {' '.join(logit_strs)}")

    scheme_label = {"mixed": "mixed (mlp+attn=Q8, rest=Q4g64)", "all-q4g64": "all Q4 g=64", "all-q8": "all Q8 g=64"}[args.scheme]
    print(f"=== Q4 Ref ({scheme_label}) ===")
    if args.scheme == 'mixed':
        model_quant = mixed_quant_model(model)
    elif args.scheme == 'all-q8':
        model_quant = all_q8_model(model)
    else:
        model_quant = q4_roundtrip_model(model)
    for i, prompt in enumerate(PROMPTS):
        prompt_ids = tok.encode(prompt)
        result = greedy_generate_transformers(model_quant, prompt_ids, max_new, old_to_new)

        if old_to_new is not None:
            gen_old, gen_new, logits = result
            logit_strs = [f"{v:.4f}" for v in logits]
            print(f"Q4_REF_{i}_old: {gen_old}")
            print(f"Q4_REF_{i}_new: {gen_new}")
            print(f"Q4_LOGITS_{i}: {' '.join(logit_strs)}")
        else:
            gen_ids, logits = result
            logit_strs = [f"{v:.4f}" for v in logits]
            print(f"Q4_REF_{i}: {gen_ids}")
            print(f"Q4_LOGITS_{i}: {' '.join(logit_strs)}")


if __name__ == '__main__':
    main()
