#!/usr/bin/env python3
"""Fine-tune ChinnaLLM from prepared masked chat tensors."""

from __future__ import annotations

import argparse
import json
import os
import re
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


class TensorDataset(Dataset):
    def __init__(self, path: Path):
        payload = torch.load(path, map_location="cpu")
        self.input_ids = payload["input_ids"]
        self.attention_mask = payload["attention_mask"]
        self.labels = payload["labels"]

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.labels[index]}


class MemmapDataset(Dataset):
    def __init__(self, data_dir: Path, split: str):
        meta = json.loads((data_dir / "dataset_meta.json").read_text())
        self.count = int(meta[f"{split}_examples"])
        self.max_length = int(meta["max_length"])
        shape = (self.count, self.max_length)
        self.input_ids = np.memmap(data_dir / f"{split}_input_ids.bin", dtype=np.uint16, mode="r", shape=shape)
        self.attention_mask = np.memmap(data_dir / f"{split}_attention_mask.bin", dtype=np.uint8, mode="r", shape=shape)
        self.labels = np.memmap(data_dir / f"{split}_labels.bin", dtype=np.int32, mode="r", shape=shape)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(np.array(self.input_ids[index], dtype=np.int64)),
            "attention_mask": torch.from_numpy(np.array(self.attention_mask[index], dtype=np.int64)),
            "labels": torch.from_numpy(np.array(self.labels[index], dtype=np.int64)),
        }


def load_dataset(data_dir: Path, split: str) -> Dataset | None:
    if (data_dir / "dataset_meta.json").exists():
        return MemmapDataset(data_dir, split)
    path = data_dir / f"{split}.pt"
    return TensorDataset(path) if path.exists() else None


def setup_distributed() -> tuple[bool, int, int]:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False, 0, 0
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, rank


def cleanup_distributed(enabled: bool) -> None:
    if not enabled or not dist.is_initialized():
        return
    try:
        dist.barrier(device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
    except Exception as exc:
        print(f"warning: distributed barrier failed during cleanup: {exc}", flush=True)
    finally:
        dist.destroy_process_group()


def pick_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def latest_checkpoint(out_dir: Path) -> Path | None:
    candidates = []
    for path in out_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "trainer_state.pt").exists():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None))[1]


def save_checkpoint(path: Path, model, tokenizer, optimizer, scheduler, step: int, args) -> None:
    path.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "args": vars(args)}, path / "trainer_state.pt")


def cosine_with_floor(optimizer, warmup_steps: int, max_steps: int, lr: float, min_lr: float):
    floor = min_lr / lr

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return floor + 0.5 * (1.0 - floor) * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device, max_batches: int = 20) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            batch = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def sample(model, tokenizer, device) -> str:
    raw = model.module if hasattr(model, "module") else model
    raw.eval()
    prompt = "System: You write simple, clear short stories for young children.\nUser: Write a short story about sharing a toy.\nAssistant: "
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = raw.generate(**inputs, max_new_tokens=80, do_sample=False, pad_token_id=0, eos_token_id=0)
    raw.train()
    return tokenizer.decode(output[0].detach().cpu().tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="out/final")
    parser.add_argument("--data-dir", default="data/chat_ft")
    parser.add_argument("--out-dir", default="out_ft")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="enable CUDA fp16 autocast")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    ddp, local_rank, rank = setup_distributed()
    is_main = rank == 0
    device = pick_device(local_rank)
    torch.manual_seed(args.seed + rank)

    try:
        data_dir = Path(args.data_dir)
        train_data = load_dataset(data_dir, "train")
        val_data = load_dataset(data_dir, "val")
        if train_data is None:
            raise FileNotFoundError(f"no train dataset found in {data_dir}")
        train_sampler = DistributedSampler(train_data, shuffle=True) if ddp else RandomSampler(train_data)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False) if val_data else None

        out_dir = Path(args.out_dir)
        resume_checkpoint = latest_checkpoint(out_dir) if args.resume else None
        load_dir = resume_checkpoint if resume_checkpoint else Path(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(load_dir).to(device)
        tokenizer = AutoTokenizer.from_pretrained(load_dir, use_fast=True)
        if ddp:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = cosine_with_floor(optimizer, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        step = 0
        if resume_checkpoint:
            state = torch.load(resume_checkpoint / "trainer_state.pt", map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            step = int(state["step"])
            if is_main:
                print(f"resuming from {resume_checkpoint} at step {step}", flush=True)

        use_amp = torch.cuda.is_available() and args.amp and not args.fp32
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        loss_curve: list[dict[str, float]] = []
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while step < args.max_steps:
            if ddp and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(step)
            for batch in train_loader:
                batch = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
                with (torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()):
                    loss = model(**batch).loss / args.grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % args.grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                step += 1
                train_loss = float(loss.detach().cpu()) * args.grad_accum
                if is_main:
                    lr = scheduler.get_last_lr()[0]
                    print(f"step {step}/{args.max_steps} train_loss={train_loss:.4f} lr={lr:.6g}", flush=True)
                    loss_curve.append({"step": step, "train_loss": train_loss, "lr": lr})
                if is_main and val_loader and step % args.eval_every == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"step {step}/{args.max_steps} val_loss={val_loss:.4f}", flush=True)
                    loss_curve[-1]["val_loss"] = val_loss
                if is_main and args.sample_every and step % args.sample_every == 0:
                    text = sample(model, tokenizer, device)
                    print(f"step {step}/{args.max_steps} sample={text!r}", flush=True)
                if is_main and step % args.save_every == 0:
                    save_checkpoint(out_dir / f"checkpoint-{step}", model, tokenizer, optimizer, scheduler, step, args)
                if step >= args.max_steps:
                    break

        if is_main:
            save_checkpoint(out_dir / f"checkpoint-{step}", model, tokenizer, optimizer, scheduler, step, args)
            (out_dir / "loss_curve.json").write_text(json.dumps(loss_curve, indent=2) + "\n", encoding="utf-8")
            print(f"wrote final checkpoint and loss curve to {out_dir}", flush=True)
    finally:
        cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
