#!/usr/bin/env python3
"""Create a small TinyStories-Instruct-style chat corpus for local dry runs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


NAMES = ["Lily", "Ben", "Mia", "Tom", "Nora", "Sam", "Ava", "Leo"]
PLACES = ["garden", "kitchen", "tree house", "beach", "library", "farm", "park", "bedroom"]
OBJECTS = ["red ball", "tiny boat", "blue kite", "lost mitten", "wooden train", "shiny shell"]
LESSONS = ["sharing", "being brave", "telling the truth", "helping a friend", "patience", "kindness"]


def make_story(rng: random.Random) -> tuple[str, str]:
    name = rng.choice(NAMES)
    friend = rng.choice([n for n in NAMES if n != name])
    place = rng.choice(PLACES)
    obj = rng.choice(OBJECTS)
    lesson = rng.choice(LESSONS)
    prompt = f"Write a short story for a young child about {lesson}. Use simple words."
    story = (
        f"{name} went to the {place} with a {obj}. {friend} wanted to play too, "
        f"but {name} held it tightly. Then {name} saw {friend} look sad. "
        f"So {name} smiled and said, 'We can play together.' They took turns, laughed, "
        f"and made a new game. At bedtime, {name} felt happy because {lesson} made the day better."
    )
    return prompt, story


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/chat_ft/dryrun.jsonl")
    parser.add_argument("--target-mb", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = int(args.target_mb * 1024 * 1024)
    written = 0
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        while written < target_bytes:
            prompt, story = make_story(rng)
            record = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": story}]}
            line = json.dumps(record, ensure_ascii=True) + "\n"
            handle.write(line)
            written += len(line.encode("utf-8"))
            count += 1
    print(json.dumps({"path": str(out), "records": count, "bytes": written}, indent=2))


if __name__ == "__main__":
    main()
