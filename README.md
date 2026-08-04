# ChinnaLLM 🐘→🐜

**A 2.7M-parameter LLM trained from scratch, small enough to tell stories on a
$4 microcontroller.**

*Chinna* (சின்ன) means **"small"** in Tamil — a small LLM, named honestly.

ChinnaLLM is a GPT-Neo-architecture model (8 layers × 128 dim, 8K custom BPE
vocab) pretrained and chat fine-tuned on 2× RTX 3090, then quantized to Q8 and
deployed on an RP2040 (264 KB RAM, no FPU) where it powers
[PicoTales](https://github.com/azhar0406/PicoTales) — a standalone
voice→story machine. The weights stream from flash via XIP; RAM holds only the
KV-cache and activations.

## What's here

| Path | What |
|---|---|
| `models/chinnallm/` | Base pretrained model (HF format) |
| `models/chinnallm-chat/` | Chat fine-tune (HF format) — the one PicoTales ships |
| `models/chinnallm_chat_q8.bin` + `_tok.bin` | Q8 export for device (SPEC_Q4_NEO format) |
| `finetune/` | Chat dataset builder + fine-tune script |
| `tools/export_q4_neo.py` | HF → device binary exporter (Q8/Q4, vocab remap) |
| `tools/hf_ref_neo.py` | Host reference decoder for parity testing |
| `tools/gen_template_ids.py` | Precomputes chat-template token IDs for firmware |
| `CHINNALLM.md` | Full model card: specs, training record, port report |
| `SPEC_Q4_NEO.md` | On-device binary format specification |

## Model card (short version)

| | |
|---|---|
| Architecture | GPT-Neo (global attention), tied embeddings, learned positions |
| Size | ~2.7M params — 8L × 128d, 16 heads, FFN 512 |
| Vocab | 8,192 custom BPE |
| Context | 512 native (capped to 88–96 on device) |
| Pretrain | 25,000 iters, 2× RTX 3090 DDP, val loss 1.63 |
| Chat FT | 4,000 steps (~262M tokens), val loss 1.49, System/User/Assistant template |
| Quantized | all-Q8, 2.7 MB on flash, bit-exact host↔device parity |

See [CHINNALLM.md](CHINNALLM.md) for the complete training record and the
RP2040 port report (including the BPE encoder bug that only surfaced on
device).

## Try it on your machine

```sh
pip install torch transformers safetensors
python tools/hf_ref_neo.py models/chinnallm-chat
```

## Reproduce the device binary

```sh
python tools/export_q4_neo.py models/chinnallm-chat \
    models/chinnallm_chat_q8.bin --all-q8
```

Then drop the `.bin` + `_tok.bin` into
[PicoTales](https://github.com/azhar0406/PicoTales)' `models/` and rebuild.

## License

MIT — see [LICENSE](LICENSE). Training/inference tooling descended in spirit
from [llama2.c](https://github.com/karpathy/llama2.c) (MIT).
