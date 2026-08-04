#!/usr/bin/env python3
"""Build ChinnaLLM chat fine-tune datasets.

Supports JSON/JSONL chat records for dry runs and raw TinyStories-Instruct text
for the full run. The full format writes disk-backed memmaps so the trainer does
not need to load the entire corpus into RAM.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from transformers import AutoTokenizer


EOS_TEXT = "<|endoftext|>"
ROLE_LABELS = {"system": "System", "user": "User", "human": "User", "assistant": "Assistant", "gpt": "Assistant", "bot": "Assistant"}
SYSTEM_PROMPT = "You write simple, clear short stories for young children."


class MemmapWriter:
    def __init__(self, out_dir: Path, split: str, max_length: int):
        self.split = split
        self.max_length = max_length
        self.count = 0
        self.input = (out_dir / f"{split}_input_ids.bin").open("wb")
        self.labels = (out_dir / f"{split}_labels.bin").open("wb")
        self.attn = (out_dir / f"{split}_attention_mask.bin").open("wb")

    def write(self, example: dict) -> None:
        np.asarray(example["input_ids"], dtype=np.uint16).tofile(self.input)
        np.asarray(example["labels"], dtype=np.int32).tofile(self.labels)
        np.asarray(example["attention_mask"], dtype=np.uint8).tofile(self.attn)
        self.count += 1

    def close(self) -> None:
        self.input.close()
        self.labels.close()
        self.attn.close()


def iter_docs(path: Path) -> Iterator[str]:
    buf = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(16 * 1024 * 1024)
            if not chunk:
                break
            buf += chunk
            parts = buf.split(EOS_TEXT)
            buf = parts.pop()
            for part in parts:
                doc = part.strip()
                if doc:
                    yield doc
        if buf.strip():
            yield buf.strip()


def parse_instruct_doc(doc: str) -> dict[str, str] | None:
    fields: dict[str, str] = {}
    story_lines: list[str] = []
    in_story = False
    for line in doc.splitlines():
        stripped = line.strip()
        if not in_story and stripped.startswith("Story:"):
            in_story = True
            rest = stripped[len("Story:"):].strip()
            if rest:
                story_lines.append(rest)
            continue
        if in_story:
            story_lines.append(line.rstrip())
            continue
        for key in ("Summary", "Words", "Features"):
            prefix = f"{key}:"
            if stripped.startswith(prefix):
                fields[key.lower()] = stripped[len(prefix):].strip()
                break
    story = "\n".join(story_lines).strip()
    if not story:
        return None
    fields["story"] = story
    return fields


def instruct_messages(fields: dict[str, str]) -> list[dict[str, str]]:
    request = ["Write a short story for a young child."]
    if fields.get("features"):
        request.append(f"Features: {fields['features']}")
    if fields.get("words"):
        request.append(f"Use these words: {fields['words']}")
    if fields.get("summary"):
        request.append(f"Summary: {fields['summary']}")
    return [
        {"role": "System", "content": SYSTEM_PROMPT},
        {"role": "User", "content": "\n".join(request)},
        {"role": "Assistant", "content": fields["story"]},
    ]


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("data", "examples", "records"):
                if isinstance(data.get(key), list):
                    return data[key]
        if isinstance(data, list):
            return data
        raise ValueError(f"unsupported JSON object in {path}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_messages(record: dict) -> list[dict[str, str]]:
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        out = []
        for message in messages:
            role = str(message.get("role") or message.get("from") or "").lower()
            content = message.get("content") or message.get("value") or message.get("text")
            if role and content is not None:
                out.append({"role": ROLE_LABELS.get(role, role.title()), "content": str(content).strip()})
        return [m for m in out if m["content"]]
    instruction = record.get("instruction") or record.get("prompt") or record.get("question")
    response = record.get("output") or record.get("response") or record.get("answer") or record.get("completion")
    if instruction is not None and response is not None:
        user = str(instruction).strip()
        if record.get("input"):
            user = f"{user}\n{str(record['input']).strip()}"
        return [{"role": "User", "content": user}, {"role": "Assistant", "content": str(response).strip()}]
    return []


def pad_example(input_ids: list[int], labels: list[int], max_length: int, eos_id: int) -> dict | None:
    if not input_ids or len(input_ids) > max_length:
        return None
    attn = [1] * len(input_ids)
    pad = max_length - len(input_ids)
    return {"input_ids": input_ids + [eos_id] * pad, "labels": labels + [-100] * pad, "attention_mask": attn + [0] * pad}


def encode_chat(tokenizer, messages: list[dict[str, str]], max_length: int, eos_id: int) -> dict | None:
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        prefix_ids = tokenizer.encode(f"{message['role']}: ", add_special_tokens=False)
        content_ids = tokenizer.encode(message["content"], add_special_tokens=False)
        suffix_ids = tokenizer.encode("\n", add_special_tokens=False)
        input_ids.extend(prefix_ids + content_ids + suffix_ids)
        if message["role"] == "Assistant":
            labels.extend([-100] * len(prefix_ids) + content_ids + suffix_ids)
        else:
            labels.extend([-100] * (len(prefix_ids) + len(content_ids) + len(suffix_ids)))
    input_ids.append(eos_id)
    labels.append(eos_id if any(label != -100 for label in labels) else -100)
    return pad_example(input_ids, labels, max_length, eos_id)


def encode_plain(tokenizer, text: str, max_length: int, eos_id: int) -> dict | None:
    ids = tokenizer.encode(text.strip(), add_special_tokens=False)[: max_length - 1] + [eos_id]
    return pad_example(ids, ids[:], max_length, eos_id)


def iter_json_records(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        yield from load_records(path)


def save_pt_split(path: Path, examples: list[dict]) -> None:
    torch.save({key: torch.stack([torch.tensor(example[key]) for example in examples]) for key in examples[0]}, path)


def build_json_dataset(args, tokenizer, eos_id: int, out_dir: Path) -> dict:
    raw = dropped_empty = dropped_long = 0
    examples = []
    for record in iter_json_records(Path(p) for p in args.input):
        raw += 1
        messages = normalize_messages(record)
        if not messages:
            dropped_empty += 1
            continue
        encoded = encode_chat(tokenizer, messages, args.max_length, eos_id)
        if encoded is None:
            dropped_long += 1
            continue
        examples.append(encoded)
    if len(examples) < 2:
        raise ValueError("need at least two encodable examples")
    random.Random(args.seed).shuffle(examples)
    val_count = max(1, int(round(len(examples) * args.val_ratio)))
    val = examples[:val_count]
    train = examples[val_count:]
    save_pt_split(out_dir / "train.pt", train)
    save_pt_split(out_dir / "val.pt", val)
    lengths = [sum(ex["attention_mask"]) for ex in examples]
    return {"format": "pt", "raw_records": raw, "kept_examples": len(examples), "train_examples": len(train), "val_examples": len(val), "dropped_empty": dropped_empty, "dropped_over_max_length": dropped_long, "lengths": lengths, "chat_examples": len(examples), "plain_examples": 0}


def build_text_dataset(args, tokenizer, eos_id: int, out_dir: Path) -> dict:
    rng = random.Random(args.seed)
    train_writer = MemmapWriter(out_dir, "train", args.max_length)
    val_writer = MemmapWriter(out_dir, "val", args.max_length)
    plain_iter = iter_docs(Path(args.plain_text)) if args.plain_text else iter(())
    raw = kept = chat_kept = plain_kept = dropped = 0
    lengths: list[int] = []

    def emit(example: dict, kind: str) -> None:
        nonlocal kept, chat_kept, plain_kept
        (val_writer if rng.random() < args.val_ratio else train_writer).write(example)
        kept += 1
        chat_kept += int(kind == "chat")
        plain_kept += int(kind == "plain")
        lengths.append(sum(example["attention_mask"]))

    try:
        for doc in iter_docs(Path(args.instruct_text)):
            raw += 1
            fields = parse_instruct_doc(doc)
            if not fields:
                dropped += 1
                continue
            encoded = encode_chat(tokenizer, instruct_messages(fields), args.max_length, eos_id)
            if encoded is None:
                dropped += 1
                continue
            emit(encoded, "chat")
            if args.plain_text and chat_kept % max(1, int(round((1 - args.plain_fraction) / args.plain_fraction))) == 0:
                try:
                    plain_doc = next(plain_iter)
                except StopIteration:
                    continue
                plain = encode_plain(tokenizer, plain_doc, args.max_length, eos_id)
                if plain is not None:
                    emit(plain, "plain")
    finally:
        train_writer.close()
        val_writer.close()

    meta = {"format": "memmap", "max_length": args.max_length, "train_examples": train_writer.count, "val_examples": val_writer.count}
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"format": "memmap", "raw_records": raw, "kept_examples": kept, "train_examples": train_writer.count, "val_examples": val_writer.count, "dropped_empty": 0, "dropped_over_max_length": dropped, "lengths": lengths, "chat_examples": chat_kept, "plain_examples": plain_kept}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default="out/final")
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--instruct-text")
    parser.add_argument("--plain-text")
    parser.add_argument("--plain-fraction", type=float, default=0.10)
    parser.add_argument("--out-dir", default="data/chat_ft")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.max_length > 512:
        raise ValueError("ChinnaLLM context is 512; use --max-length <= 512")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    if eos_id != 0:
        raise ValueError(f"expected ChinnaLLM eos_token_id 0, got {eos_id}")

    if args.instruct_text:
        summary = build_text_dataset(args, tokenizer, eos_id, out_dir)
    elif args.input:
        summary = build_json_dataset(args, tokenizer, eos_id, out_dir)
    else:
        raise ValueError("provide --input or --instruct-text")

    lengths = summary.pop("lengths")
    stats = {
        **summary,
        "max_length": args.max_length,
        "eos_token_id": eos_id,
        "min_tokens": min(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 2),
        "max_tokens": max(lengths),
        "template": "System: <text>\\nUser: <text>\\nAssistant: <text>\\n<eos>",
        "template_token_ids": {
            "System": tokenizer.encode("System: ", add_special_tokens=False),
            "User": tokenizer.encode("User: ", add_special_tokens=False),
            "Assistant": tokenizer.encode("Assistant: ", add_special_tokens=False),
            "newline": tokenizer.encode("\n", add_special_tokens=False),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
