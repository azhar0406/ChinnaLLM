#!/usr/bin/env python3
"""
export_q4_neo.py - Convert GPT-Neo TinyStories-Instruct pytorch_model.bin to SPEC_Q4_NEO.

Usage:
    python3 tools/export_q4_neo.py <model_dir> <output_path> [--seq-len N] [--keep keep.txt]
"""

import argparse
import json
import os
import struct

import numpy as np
import torch


MAGIC = 0x51344C4D
VERSION = 1
VERSION_EXTRA = 4
VERSION_EXTRA_ALL_Q8 = 5
GROUP_SIZE = 64
FLAG_TIED = 1 << 0
FLAG_GPTNEO = 1 << 2
FLAG_ALL_Q8 = 1 << 3


def fp16_to_uint16(val):
    return int(np.float16(val).view(np.uint16))


def fp16_ceil_scale(val):
    # NOTE: returns (uint16_bits, float32) tuple — differs from hf_ref_neo.py single-float variant
    half = np.float16(val)
    if float(half) < float(val):
        half = np.nextafter(half, np.float16(np.inf), dtype=np.float16)
    return int(half.view(np.uint16)), float(half.astype(np.float32))


def pad_in_ch(arr, group_size=GROUP_SIZE):
    if arr.shape[1] % group_size == 0:
        return arr
    pad = group_size - (arr.shape[1] % group_size)
    return np.pad(arr, ((0, 0), (0, pad)), mode="constant", constant_values=0)


def quantize_q4_spec(arr, group_size=GROUP_SIZE):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Q4 tensors must be 2D, got shape {arr.shape}")
    out_dim, in_dim = arr.shape
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim {in_dim} must be divisible by group_size {group_size}")

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
            scale_bits, scale_f32 = fp16_ceil_scale(amax / 7.0)
            scales[o, g] = scale_bits
            q = np.clip(np.round(group_vals / scale_f32), -7, 7).astype(np.int8)
            for j in range(group_size):
                nibble = (int(q[j]) + 8) & 0xF
                bi = gs // 2 + j // 2
                if j % 2 == 0:
                    packed[o, bi] = nibble
                else:
                    packed[o, bi] |= nibble << 4

    return packed, scales.flatten()


def dequantize_q4_spec(packed, scales, shape, group_size=GROUP_SIZE):
    out_dim, in_dim = shape
    scales = np.asarray(scales, dtype=np.uint16).reshape(out_dim, in_dim // group_size)
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    scale_f32 = scales.view(np.float16).astype(np.float32)
    for o in range(out_dim):
        for g in range(in_dim // group_size):
            gs = g * group_size
            for j in range(group_size):
                b = packed[o, (gs + j) // 2]
                nibble = (b & 0xF) if j % 2 == 0 else (b >> 4)
                out[o, gs + j] = (int(nibble) - 8) * scale_f32[o, g]
    return out


def quantize_q8_spec(arr, group_size=GROUP_SIZE):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Q8 tensors must be 2D, got shape {arr.shape}")
    out_dim, in_dim = arr.shape
    if in_dim % group_size != 0:
        raise ValueError(f"in_dim {in_dim} must be divisible by group_size {group_size}")

    n_groups = in_dim // group_size
    qbytes = np.zeros((out_dim, in_dim), dtype=np.int8)
    scales = np.zeros((out_dim, n_groups), dtype=np.uint16)

    for o in range(out_dim):
        for g in range(n_groups):
            gs = g * group_size
            ge = gs + group_size
            group_vals = arr[o, gs:ge]
            amax = float(np.abs(group_vals).max())
            if amax < 1e-12:
                amax = 1.0
            scale_bits, scale_f32 = fp16_ceil_scale(amax / 127.0)
            scales[o, g] = scale_bits
            qbytes[o, gs:ge] = np.clip(np.round(group_vals / scale_f32), -127, 127).astype(np.int8)

    return qbytes, scales.flatten()


def dequantize_q8_spec(qbytes, scales, shape, group_size=GROUP_SIZE):
    out_dim, in_dim = shape
    scales = np.asarray(scales, dtype=np.uint16).reshape(out_dim, in_dim // group_size)
    qbytes = np.asarray(qbytes, dtype=np.int8).reshape(out_dim, in_dim)
    out = np.zeros((out_dim, in_dim), dtype=np.float32)
    scale_f32 = scales.view(np.float16).astype(np.float32)
    for o in range(out_dim):
        for g in range(in_dim // group_size):
            gs = g * group_size
            ge = gs + group_size
            out[o, gs:ge] = qbytes[o, gs:ge].astype(np.float32) * scale_f32[o, g]
    return out


def load_weights(model_dir):
    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    st_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(st_path):
        import safetensors.torch
        state = safetensors.torch.load_file(st_path, device="cpu")
        print(f"Loaded safetensors: {st_path}")
    elif os.path.exists(pt_path):
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        print(f"Loaded pytorch_model.bin: {pt_path}")
    else:
        raise FileNotFoundError(f"No model weights found at {pt_path} or {st_path}")
    weights = {}
    skipped = []
    for key, tensor in state.items():
        if key.endswith("attn.attention.bias") or key.endswith("attn.attention.masked_bias"):
            skipped.append(key)
            continue
        weights[key] = tensor.detach().cpu().float().numpy()
    return weights, skipped, list(state.keys())


def require(weights, key, shape=None):
    if key not in weights:
        available = "\n  ".join(sorted(weights))
        raise KeyError(f"Missing required key {key}. Available learned keys:\n  {available}")
    arr = weights[key]
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{key} shape {arr.shape} != {shape}")
    return arr.astype(np.float32, copy=False)


def read_keep(path, vocab_size):
    if path is None:
        return list(range(vocab_size))
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    ids = []
    if path.endswith(".json"):
        data = json.loads(content)
        if isinstance(data, dict):
            data = data.get("token_ids")
        if not isinstance(data, list):
            raise ValueError("JSON keep-set must be a list or an object with token_ids")
        ids = [int(x) for x in data]
    else:
        for field in content.split():
            tok_id = int(field)
            ids.append(tok_id)
    if not ids:
        raise ValueError("--keep file did not contain any token ids")
    for tok_id in ids:
        if tok_id < 0 or tok_id >= vocab_size:
            raise ValueError(f"Token id {tok_id} outside vocab range 0..{vocab_size - 1}")
    return sorted(set(ids))


def output_stem(output_path):
    return output_path[:-4] if output_path.endswith(".bin") else output_path


def load_id_to_token(model_dir):
    vocab_path = os.path.join(model_dir, "vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return {int(v): k for k, v in vocab.items()}


def write_tokenizer(model_dir, tok_path, new_to_old):
    with open(os.path.join(model_dir, "vocab.json"), "r", encoding="utf-8") as f:
        token_to_id = {k: int(v) for k, v in json.load(f).items()}
    id_to_token = {v: k for k, v in token_to_id.items()}
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    encoded = []
    for old_id in new_to_old:
        token = id_to_token.get(old_id, "")
        encoded.append(token.encode("utf-8"))
    max_len = max((len(t) for t in encoded), default=0)
    offsets = []
    cursor = 0
    for token in encoded:
        offsets.append(cursor)
        cursor += len(token)

    with open(tok_path, "wb") as f:
        f.write(struct.pack("<I", len(encoded)))
        f.write(struct.pack("<I", max_len))
        for token in encoded:
            f.write(struct.pack("<H", len(token)))
        for token in encoded:
            f.write(token)
        # Zero-pad so the appended u32 offsets table starts at a file offset
        # that is a multiple of 4 (enables XIP zero-copy on device).
        pad = (-(8 + 2 * len(encoded) + cursor)) % 4
        f.write(b"\x00" * pad)
        for off in offsets:
            f.write(struct.pack("<I", off))

        merges_path = os.path.join(model_dir, "merges.txt")
        if os.path.exists(merges_path):
            triples = []
            with open(merges_path, "r", encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    left, right = parts
                    merged = left + right
                    if left in token_to_id and right in token_to_id and merged in token_to_id:
                        lo, ro, oo = token_to_id[left], token_to_id[right], token_to_id[merged]
                        if lo in old_to_new and ro in old_to_new and oo in old_to_new:
                            triples.append((old_to_new[lo], old_to_new[ro], old_to_new[oo]))
            # Optional v2 section, naturally 4-byte aligned after the u32 offsets table.
            f.write(struct.pack("<II", 0x31455042, len(triples)))  # "BPE1"
            for col in range(3):
                for triple in triples:
                    f.write(struct.pack("<H", triple[col]))


def write_remap(remap_path, new_to_old, eos_new, eos_old):
    old_to_new = {str(old): new for new, old in enumerate(new_to_old)}
    with open(remap_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "old_to_new": old_to_new,
                "new_to_old": new_to_old,
                "eos_token_id": eos_new,
                "original_eos_token_id": eos_old,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


def append_fp32(parts, arr):
    parts.append(np.asarray(arr, dtype=np.float32).tobytes())


def export_q4_neo(model_dir, output_path, seq_len=160, keep_path=None, all_q8=False):
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    if config.get("model_type") != "gpt_neo":
        raise ValueError(f"Expected model_type gpt_neo, got {config.get('model_type')}")
    if config.get("activation_function") != "gelu_new":
        raise ValueError(f"Expected gelu_new activation, got {config.get('activation_function')}")

    dim = int(config["hidden_size"])
    hidden_dim = 4 * dim
    n_layers = int(config["num_layers"])
    n_heads = int(config["num_heads"])
    n_kv_heads = n_heads
    head_dim = dim // n_heads
    vocab_size_full = int(config["vocab_size"])
    eos_old = int(config.get("eos_token_id", 50256))
    wpe_rows = int(seq_len)
    max_pos = int(config["max_position_embeddings"])
    if wpe_rows > max_pos:
        raise ValueError(f"seq_len {wpe_rows} exceeds max_position_embeddings {max_pos}")

    new_to_old = read_keep(keep_path, vocab_size_full)
    if eos_old not in new_to_old:
        raise ValueError(f"EOS token id {eos_old} must be present in --keep set")
    old_to_new = {old: new for new, old in enumerate(new_to_old)}
    eos_new = old_to_new[eos_old]
    vocab_size = len(new_to_old)

    weights, skipped, _ = load_weights(model_dir)
    version_extra = VERSION_EXTRA_ALL_Q8 if all_q8 else VERSION_EXTRA
    flags = FLAG_TIED | FLAG_GPTNEO
    if all_q8:
        flags |= FLAG_ALL_Q8

    print(f"Exporting GPT-Neo {model_dir} -> {output_path}")
    if all_q8:
        print(f"  Mode: ALL-Q8 (wte, wpe, attn, mlp all Q8g64)")
    print(f"  dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  vocab={vocab_size} of {vocab_size_full}, seq_len={seq_len}, wpe_rows={wpe_rows}")
    print(f"  eos old={eos_old}, eos remapped={eos_new}, group_size={GROUP_SIZE}")
    print(f"  skipped non-learned buffers={len(skipped)}")

    parts = []
    header = bytearray(64)
    struct.pack_into("<I", header, 0, MAGIC)
    struct.pack_into("<i", header, 4, VERSION)
    struct.pack_into("<i", header, 8, dim)
    struct.pack_into("<i", header, 12, hidden_dim)
    struct.pack_into("<i", header, 16, n_layers)
    struct.pack_into("<i", header, 20, n_heads)
    struct.pack_into("<i", header, 24, n_kv_heads)
    struct.pack_into("<i", header, 28, vocab_size)
    struct.pack_into("<i", header, 32, seq_len)
    struct.pack_into("<i", header, 36, GROUP_SIZE)
    struct.pack_into("<I", header, 40, flags)
    struct.pack_into("<f", header, 44, 0.0)
    struct.pack_into("<f", header, 48, float(config.get("layer_norm_epsilon", 1e-5)))
    struct.pack_into("<i", header, 52, version_extra)
    struct.pack_into("<i", header, 56, eos_new)
    struct.pack_into("<i", header, 60, wpe_rows)
    parts.append(bytes(header))

    for layer in range(n_layers):
        append_fp32(parts, require(weights, f"transformer.h.{layer}.ln_1.weight", (dim,)))
        append_fp32(parts, require(weights, f"transformer.h.{layer}.ln_1.bias", (dim,)))
    for layer in range(n_layers):
        append_fp32(parts, require(weights, f"transformer.h.{layer}.ln_2.weight", (dim,)))
        append_fp32(parts, require(weights, f"transformer.h.{layer}.ln_2.bias", (dim,)))
    append_fp32(parts, require(weights, "transformer.ln_f.weight", (dim,)))
    append_fp32(parts, require(weights, "transformer.ln_f.bias", (dim,)))
    for layer in range(n_layers):
        append_fp32(parts, require(weights, f"transformer.h.{layer}.attn.attention.out_proj.bias", (dim,)))
    for layer in range(n_layers):
        append_fp32(parts, require(weights, f"transformer.h.{layer}.mlp.c_fc.bias", (hidden_dim,)))
    for layer in range(n_layers):
        append_fp32(parts, require(weights, f"transformer.h.{layer}.mlp.c_proj.bias", (dim,)))

    q4_bytes = 0
    q8_bytes = 0

    def quantize_and_append(name, arr, expected_shape, bits):
        nonlocal q4_bytes, q8_bytes
        if arr.shape != expected_shape:
            raise ValueError(f"{name} shape {arr.shape} != {expected_shape}")
        padded = pad_in_ch(arr)
        if bits == 4:
            qdata, scales = quantize_q4_spec(padded)
            label = "Q4"
            q4_bytes += qdata.nbytes + scales.nbytes
        elif bits == 8:
            qdata, scales = quantize_q8_spec(padded)
            label = "Q8"
            q8_bytes += qdata.nbytes + scales.nbytes
        else:
            raise ValueError(f"unsupported quantization width {bits}")
        parts.append(qdata.tobytes())
        parts.append(scales.astype(np.uint16).tobytes())
        size = qdata.nbytes + scales.nbytes
        print(f"  {label} {name}: shape={arr.shape}, padded_in={padded.shape[1]}, bytes={size:,}")
        return size

    wte = require(weights, "transformer.wte.weight", (vocab_size_full, dim))[new_to_old, :]
    quantize_and_append("wte", wte, (vocab_size, dim), 8 if all_q8 else 4)

    wpe_full = require(weights, "transformer.wpe.weight", (max_pos, dim))
    quantize_and_append("wpe", wpe_full[:wpe_rows, :], (wpe_rows, dim), 8 if all_q8 else 4)

    tensor_classes = [
        ("wq", "attn.attention.q_proj.weight", (dim, dim)),
        ("wk", "attn.attention.k_proj.weight", (dim, dim)),
        ("wv", "attn.attention.v_proj.weight", (dim, dim)),
        ("wo", "attn.attention.out_proj.weight", (dim, dim)),
        ("c_fc", "mlp.c_fc.weight", (hidden_dim, dim)),
        ("c_proj", "mlp.c_proj.weight", (dim, hidden_dim)),
    ]
    for label, suffix, shape in tensor_classes:
        for layer in range(n_layers):
            key = f"transformer.h.{layer}.{suffix}"
            quantize_and_append(f"{label}.{layer}", require(weights, key, shape), shape, 8)

    data = b"".join(parts)
    with open(output_path, "wb") as f:
        f.write(data)

    stem = output_stem(output_path)
    remap_path = f"{stem}_remap.json"
    tok_path = f"{stem}_tok.bin"
    write_remap(remap_path, new_to_old, eos_new, eos_old)
    write_tokenizer(model_dir, tok_path, new_to_old)

    fp32_values = (
        (4 * n_layers * dim)  # ln_1/ln_2 weights and biases
        + (2 * dim)  # ln_f weight and bias
        + (n_layers * dim)  # out_proj.bias
        + (n_layers * hidden_dim)  # c_fc.bias
        + (n_layers * dim)  # c_proj.bias
    )
    fp32_size = fp32_values * 4
    total = len(data)
    print(f"\n  Written {output_path}")
    print(f"  Total: {total:,} B ({total / 1024 / 1024:.2f} MB)")
    print(f"  Header: 64 B")
    print(f"  fp32 section: {fp32_size:,} B")
    print(f"  q4 weights: {q4_bytes:,} B ({q4_bytes / 1024 / 1024:.2f} MB)")
    print(f"  q8 weights: {q8_bytes:,} B ({q8_bytes / 1024 / 1024:.2f} MB)")
    print(f"  quantized weights: {q4_bytes + q8_bytes:,} B ({(q4_bytes + q8_bytes) / 1024 / 1024:.2f} MB)")
    print(f"  Remap: {remap_path} ({os.path.getsize(remap_path):,} B)")
    print(f"  Tokenizer: {tok_path} ({os.path.getsize(tok_path):,} B)")
    return total


def self_test():
    rng = np.random.default_rng(1234)
    for shape in [(3, 64), (5, 128), (7, 192)]:
        arr = rng.normal(0.0, 0.2, shape).astype(np.float32)
        packed, scales = quantize_q4_spec(arr)
        out = dequantize_q4_spec(packed, scales, shape)
        scale_f32 = scales.view(np.float16).astype(np.float32)
        limit = float(scale_f32.max() / 2.0 + 1e-6)
        err = float(np.max(np.abs(arr - out)))
        print(f"self-test shape={shape} max_err={err:.8f} scale/2_limit={limit:.8f}")
        if err > limit:
            raise AssertionError(f"max err {err} > scale/2 {limit}")
        qbytes, scales = quantize_q8_spec(arr)
        out = dequantize_q8_spec(qbytes, scales, shape)
        scale_f32 = scales.view(np.float16).astype(np.float32)
        limit = float(scale_f32.max() / 2.0 + 1e-6)
        err = float(np.max(np.abs(arr - out)))
        print(f"self-test-q8 shape={shape} max_err={err:.8f} scale/2_limit={limit:.8f}")
        if err > limit:
            raise AssertionError(f"q8 max err {err} > scale/2 {limit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GPT-Neo pytorch_model.bin to SPEC_Q4_NEO")
    parser.add_argument("model_dir", nargs="?", help="Hugging Face model directory")
    parser.add_argument("output_path", nargs="?", help="Output SPEC_Q4_NEO binary")
    parser.add_argument("--seq-len", type=int, default=160, help="Firmware sequence length / wpe rows")
    parser.add_argument("--keep", help="Optional whitespace-delimited token id keep-set")
    parser.add_argument("--all-q8", action="store_true", help="Use Q8 for ALL tensors (wte, wpe, attn, mlp)")
    parser.add_argument("--self-test", action="store_true", help="Run Q4 dequant spot-check and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        if not args.model_dir or not args.output_path:
            parser.error("model_dir and output_path are required unless --self-test is used")
        export_q4_neo(args.model_dir, args.output_path, args.seq_len, args.keep, args.all_q8)
