# SPEC_Q4_NEO — Extension for GPT-Neo TinyStories-Instruct

This document extends `SPEC_Q4.md` v1 for Hugging Face `GPTNeoForCausalLM` checkpoints exported from `pytorch_model.bin`.

## File Layout

```
[header 64 bytes][fp32 section][mixed Q4/Q8 section]
```

The first 52 header bytes remain compatible with `struct.unpack('<Ii8iIff')` and the firmware CMake guard: `magic = 0x51344C4D`, `version = 1`.

## Header

| Offset | Type | Field |
|--------|------|-------|
| 0  | u32 | magic = `0x51344C4D` |
| 4  | i32 | version = `1` |
| 8  | i32 | dim = `hidden_size` |
| 12 | i32 | hidden_dim = `4 * hidden_size` because GPT-Neo config has `intermediate_size: null` |
| 16 | i32 | n_layers = `num_layers` |
| 20 | i32 | n_heads = `num_heads` |
| 24 | i32 | n_kv_heads = `n_heads` |
| 28 | i32 | vocab_size = emitted vocabulary rows after optional pruning |
| 32 | i32 | seq_len = device context length |
| 36 | i32 | group_size = `64` |
| 40 | u32 | flags: bit0 tied embeddings = `1`, bit2 GPT-Neo = `1` |
| 44 | f32 | rope_theta = `0.0` because GPT-Neo uses learned position embeddings |
| 48 | f32 | rms_norm_eps carries `layer_norm_epsilon` |
| 52 | i32 | version_extra = `4` for GPT-Neo mixed Q4/Q8 |
| 56 | i32 | eos_token_id after remap |
| 60 | i32 | wpe_rows, normally equal to `seq_len` |

## fp32 Section

All entries are raw little-endian fp32 arrays.

Order:

1. Per layer from `0..n_layers-1`: `ln_1.weight[dim]`, then `ln_1.bias[dim]`
2. Per layer from `0..n_layers-1`: `ln_2.weight[dim]`, then `ln_2.bias[dim]`
3. Final norm: `ln_f.weight[dim]`, then `ln_f.bias[dim]`
4. Per layer from `0..n_layers-1`: `attn.attention.out_proj.bias[dim]`
5. Per layer from `0..n_layers-1`: `mlp.c_fc.bias[hidden_dim]`
6. Per layer from `0..n_layers-1`: `mlp.c_proj.bias[dim]`

## Mixed Q4/Q8 Section

Rows are `[out_ch, in_ch]`; pad `in_ch` to a multiple of 64 with zeros before quantization. No per-tensor metadata is stored; the runner derives widths from this fixed layout.

Q4g64 encoding:

1. `scale = max(abs(w)) / 7.0` per 64-value group
2. `q = clip(round(w / scale), -7, 7)`
3. Store packed unsigned nibbles `q + 8`, then fp16 scales

Q8g64 encoding:

1. `scale = max(abs(w)) / 127.0` per 64-value group
2. `q = clip(round(w / scale), -127, 127)`
3. Store raw int8 bytes, then fp16 scales

Byte formulas, where `pin = ceil(in_ch / 64) * 64`:

1. Q4 tensor bytes = `out_ch * pin / 2 + out_ch * (pin / 64) * 2`
2. Q8 tensor bytes = `out_ch * pin + out_ch * (pin / 64) * 2`

Order:

1. Q4 `wte`: `transformer.wte.weight[vocab_size, dim]`, optionally pruned and remapped
2. Q4 `wpe`: `transformer.wpe.weight[wpe_rows, dim]`, truncated to `wpe_rows = seq_len`
3. Q8 per layer class order: `wq`, `wk`, `wv`, `wo`, `c_fc`, `c_proj`
4. Within each class, layers are emitted from `0..n_layers-1`

The Q8 classes use these shapes:

1. `wq`, `wk`, `wv`, `wo`: `[dim, dim]`
2. `c_fc`: `[hidden_dim, dim]`
3. `c_proj`: `[dim, hidden_dim]`

## Byte Accounting

Header bytes: `64`.

fp32 section bytes: `(6 * n_layers * dim + n_layers * hidden_dim + 2 * dim) * 4`.

Quantized section bytes:

1. `wte_q4 = vocab_size * pad64(dim) / 2 + vocab_size * (pad64(dim) / 64) * 2`
2. `wpe_q4 = wpe_rows * pad64(dim) / 2 + wpe_rows * (pad64(dim) / 64) * 2`
3. `attn_q8 = 4 * n_layers * (dim * pad64(dim) + dim * (pad64(dim) / 64) * 2)`
4. `c_fc_q8 = n_layers * (hidden_dim * pad64(dim) + hidden_dim * (pad64(dim) / 64) * 2)`
5. `c_proj_q8 = n_layers * (dim * pad64(hidden_dim) + dim * (pad64(hidden_dim) / 64) * 2)`

Final K=16,384 production exports:

| Model | seq/wpe_rows | Header | fp32 | Q4 wte | Q4 wpe | Q8 attn | Q8 c_fc | Q8 c_proj | Total |
|-------|--------------|--------|------|--------|--------|---------|---------|-----------|-------|
| 3M (`dim=128`, `hidden_dim=512`, `layers=8`) | 64 | 64 | 41,984 | 1,114,112 | 4,352 | 540,672 | 540,672 | 540,672 | 2,782,528 |
| 1M (`dim=64`, `hidden_dim=256`, `layers=8`) | 160 | 64 | 20,992 | 557,056 | 5,440 | 135,168 | 135,168 | 135,168 | 989,056 |

GPT-Neo state dict keys:

1. `transformer.wte.weight`
2. `transformer.wpe.weight`
3. `transformer.h.{l}.ln_1.{weight,bias}`
4. `transformer.h.{l}.attn.attention.{q,k,v}_proj.weight`
5. `transformer.h.{l}.attn.attention.out_proj.{weight,bias}`
6. `transformer.h.{l}.ln_2.{weight,bias}`
7. `transformer.h.{l}.mlp.c_fc.{weight,bias}`
8. `transformer.h.{l}.mlp.c_proj.{weight,bias}`
9. `transformer.ln_f.{weight,bias}`

Non-learned buffers `attn.attention.bias` and `attn.attention.masked_bias` are ignored.

## Vocab Pruning

`tools/export_q4_neo.py --keep keep.txt` accepts one old token id per whitespace-delimited field. JSON keep sets are also accepted as either a list or an object with `token_ids`. The exporter sorts unique ids, requires the original EOS id to be present, and remaps old ids to dense new ids by sorted order.

Sidecars:

1. `<output>_remap.json`: `{old_to_new, new_to_old, eos_token_id, original_eos_token_id}`
2. `<output>_tok.bin`: compact tokenizer using the emitted dense vocabulary order

Tokenizer sidecar format:

```
u32 vocab_size
u32 max_token_len
u16 token_len[vocab_size]
u8 token_bytes[sum(token_len)]
u8 pad[(-(8 + 2*vocab_size + sum(token_len))) mod 4]  // zeros
u32 token_offset[vocab_size]
```

Offsets are byte offsets from the start of `token_bytes`. The trailing offset table is appended after the existing compact tokenizer payload so older host tools that only read lengths and bytes can continue to parse the prefix. Zero padding is inserted after `token_bytes` so the offsets table begins at a file offset that is a multiple of 4; readers locate it by rounding `8 + 2*vocab_size + sum(token_len)` up to the next multiple of 4. This lets the device map `token_len` (2-aligned at offset 8) and `token_offset` (4-aligned) directly from XIP flash with zero SRAM copies.

Without `--keep`, the exporter emits the full vocabulary for host-side verification and still writes identity remap metadata.
