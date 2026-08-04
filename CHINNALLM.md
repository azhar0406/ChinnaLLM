# ChinnaLLM-2.7M — Model Card & RP2040 Port Report

## Model Specifications

| Parameter         | Value                         |
|-------------------|-------------------------------|
| Architecture      | GPT-Neo (global attention)    |
| Layers            | 8                             |
| Hidden dim (d)    | 128                           |
| Attention heads   | 16                            |
| FFN intermediate  | 512                           |
| Vocab size        | 8,192 (custom 8K BPE)         |
| Context length    | 512 (native), 96 (device cap) |
| Position encoding | Learned (wpe)                 |
| EOS token         | 0 (`<|endoftext|>` tied)      |
| Tie embeddings    | Yes                           |
| Activation        | GELU (new)                    |
| Total parameters  | ~2.7M                         |

## Training Record

### Pretrain

| Item          | Value                      |
|---------------|----------------------------|
| Hardware      | 2x NVIDIA RTX 3090 (DDP)   |
| Iterations    | 25,000                     |
| Final val loss| 1.6316                     |
| Dataset       | Custom 8K BPE pretraining corpus |
| Checkpoint    | `models/chinnallm/`        |

### Chat Fine-Tune

| Item          | Value                                          |
|---------------|-------------------------------------------------|
| Hardware      | 2x NVIDIA RTX 3090 (DDP)                        |
| Steps         | 4,000 (~262M tokens budget)                     |
| Final val loss| 1.4928                                         |
| Warmup        | 100 steps                                       |
| Learning rate | 1e-4 cosine decay to 1e-5                       |
| Batch size    | 64 per GPU                                      |
| Dataset size  | 2.63M examples (2.58M train / 52.5K val)        |
| Mix           | ~90% chat-formatted / ~10% plain pretrain-style |
| Source        | TinyStories-Instruct + TinyStories V2           |
| Template      | `System: <text>\nUser: <text>\nAssistant: <text>\n` |
| EOS masking   | Assistant spans supervised; System/User masked  |
| Checkpoint    | `models/chinnallm-chat/`                        |

### Loss Curve (chat fine-tune)

| Step  | Train Loss | Val Loss | LR      |
|-------|-----------|----------|---------|
| 1     | 1.7591    | —        | 2e-6    |
| 250   | —         | 1.5355   | —       |
| 1000  | 1.6293    | 1.5218   | 8.868e-5 |
| 2000  | 1.4482    | 1.5089   | 5.681e-5 |
| 3000  | 1.4908    | 1.4976   | 2.383e-5 |
| 3500  | 1.5166    | 1.4947   | 1.360e-5 |
| 4000  | 1.5393    | 1.4928   | 1e-5    |

## Template Token IDs (ChinnaLLM tokenizer)

| Element      | Token IDs                                  |
|--------------|--------------------------------------------|
| `System: `   | `[51, 89, 375, 952, 26, 221]`              |
| `User: `     | `[53, 2759, 26, 221]`                      |
| `Assistant: `| `[33, 577, 651, 701, 26, 221]`             |
| `\n`         | `[199]`                                    |
| `<eos>`      | `[0]`                                      |

## Export & Quantization

| Item                | Value                                  |
|---------------------|----------------------------------------|
| Scheme              | all-Q8 (version_extra=5)               |
| Group size          | 64 (Q8g64 symmetric per-group)         |
| Scale encoding      | fp16 ceil                               |
| Base model binary   | `models/chinnallm_q8.bin` (2,812,992 B) |
| Chat model binary   | `models/chinnallm_chat_q8.bin` (2,812,992 B) |
| Tokenizer sidecar   | `*_q8_tok.bin` (149,538 B each)        |
| Remap sidecar       | `*_q8_remap.json` (226,144 B each)     |
| Seq cap (device)    | 96 (`NEO_SEQ_CAP=96`)                  |

### Device Memory Budget (RP2040, 264 KB SRAM)

| Region      | Size                  |
|-------------|-----------------------|
| Heap        | 232,456 bytes         |
| BSS         | 4,520 bytes (4.4 KB)  |
| **Total**   | **236,976 bytes**     |
| Free heap   | 33,360 bytes (~32.6 KB) |

KV cache under seq-cap 96 shrinks from the native 512-token buffer (1024 KB would overflow) to the runtime-clamped size (~227 KB KV at cap=96 with int8-KV, plus runstate overhead).

### Flash Footprint

| Component      | Size              |
|----------------|-------------------|
| Model weights  | 2.81 MB           |
| Tokenizer      | ~100 KB           |
| Firmware + SDK | ~190 KB           |
| **Total UF2**  | **~3.0 MB**       |
| Flash budget   | 4 MB (W25Q32)     |
| Margin         | ~1 MB free        |

## Validation Chain

### 1. ID-Level Token-Exact (fp32 KV)

Host runner (`build/runq4_neo_chinnallm --kv-f32`) vs Python all-Q8 HF reference:

| Prompt        | Tokens | Edit Distance | Diverge At | Result |
|---------------|--------|---------------|------------|--------|
| Brave cat     | 30/30  | 0             | —          | PASS   |
| Happy dog     | 30/30  | 0             | —          | PASS   |
| Two friends   | 30/30  | 0             | —          | PASS   |

**3/3 token-exact. Weight dequant is correct.**

### 2. int8-KV (device default, fp16 logits)

| Prompt        | Tokens | Edit Distance | Diverge At | Result      |
|---------------|--------|---------------|------------|-------------|
| Brave cat     | 30/30  | 0             | —          | PASS        |
| Happy dog     | —      | —             | step 11    | **DIVERGE** |
| Two friends   | 30/30  | 0             | —          | PASS        |

**2/3 exact in device mode.** The divergence on 1/3 prompts is expected and accepted as int8 KV cache quantization drift.

### 3. Encoder Parity (Merge-Order BPE Fix)

A BPE encoder parity bug was discovered during chat-model validation: the host/firmware BPE encoder used longest-match segmentation, while the HF reference uses merge-order BPE. This caused prompt tokenization mismatches (e.g., `Write` → token 82 `r` in HF vs token 321 `ri` in host; `Assistant:` → `[33,577,651,701,26]` vs `[968,83,651,701,26]`).

**Fix**: Reimplemented the firmware BPE encoder with merge-rank ordering (matching HuggingFace `tokenizers` library behavior). The fix landed in `firmware/runq4_neo_host.c`, sidecar bumped to BPE1 format with 4-byte XIP-aligned merge tables.

**Parity battery**: 20/20 prompts — 100% ID sequence match between HF tokenizer and host encoder.

### 4. Text-Prompt Token-Exact (with Encoder Fix)

After the encoder fix, both base and chat Q8 binaries pass text-prompt token-exact validation: same text → same token IDs → same generated tokens (30 steps, 0 edit distance on all prompts).

## Known Limits

| Limitation            | Detail                                                     |
|-----------------------|------------------------------------------------------------|
| Instruction adherence | Loose at 2.7M params — model may drift from prompt intent |
| int8-KV drift         | 1/3 prompts diverge under device default (accepted)        |
| Inference speed       | ~0.1–0.15 tok/s expected on RP2040 (no FPU, 200 MHz)      |
| Context window        | 96 tokens (device cap); native model supports 512          |
| Tokenizer parity      | Now 100% after merge-order BPE fix; regression suite added |

## On-Device Acceptance

Measured on Shrike-lite RP2040 (200 MHz overclock) with chat UF2 via `/dev/cu.usbmodem2101`:

| Metric                   | Value                                     |
|--------------------------|-------------------------------------------|
| tok/s                    | **0.163 tok/s**                           |
| Reply time (58 tokens)   | ~356.8s (~6 min)                          |
| Heap used                | 232,448 bytes                             |
| Heap gate                | 240,000 bytes                             |
| BSS                      | ~4,520 bytes                              |
| Physical SRAM headroom   | ~33.4 KB (270,336 − 232,448 − 4,520)      |
| Context behavior         | `model_clear_kv()` called at each turn (fresh KV/window per turn) |
| `/reset` behavior        | Clears Reply/generated counters, words, temp state |
| Boot banner              | Not captured (capture started after reboot) |

Numbers supplied by worker agent during port flash phase.

## Known Issues

- **BPE1 sidecar alignment**: When `merge_count` is odd (e.g., 7935), the right merge column is 2-byte aligned rather than 4-byte aligned. This works correctly on the RP2040 (16-bit reads supported), but the right column should be padded to an even count (i.e., the next multiple of 2) in a future exporter revision for strict 4-byte alignment.

## Files Inventory

| File                                    | Purpose                          |
|-----------------------------------------|----------------------------------|
| `models/chinnallm/`                     | Base pretrained checkpoint (HF)  |
| `models/chinnallm-chat/`                | Chat fine-tuned checkpoint (HF)  |
| `models/chinnallm_q8.bin`               | Base all-Q8 export               |
| `models/chinnallm_q8_tok.bin`           | Base sidecar tokenizer           |
| `models/chinnallm_q8_remap.json`        | Base vocab remap                 |
| `models/chinnallm_chat_q8.bin`          | Chat all-Q8 export               |
| `models/chinnallm_chat_q8_tok.bin`      | Chat sidecar tokenizer           |
| `models/chinnallm_chat_q8_remap.json`   | Chat vocab remap                 |
| `tools/export_q4_neo.py`                | Model exporter (supports all-Q8) |
| `tools/hf_ref_neo.py`                   | HF reference validation          |
| `firmware/runq4_neo_host.c`             | Host/firmware inference engine   |
| `firmware/main_chat_neo.c`              | Device chat REPL firmware        |
| `training/build_chinnallm_chat_dataset.py` | Chat dataset builder          |
| `training/finetune_chinnallm.py`        | Fine-tune training script        |
| `finetune/`                             | Staging: server-side run artifacts |
