#!/usr/bin/env python3
"""Generate Semantic Part Decomposition prompts

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FIG1_TEMPLATE = "Decompose category [{category}] into semantic parts decoupling"


def load_classes(path: Path) -> list[str]:
    classes: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            classes.append(line)
    return classes


def build_records(classes: list[str], k: int, template: str) -> list[dict[str, object]]:
    records = []
    for class_name in classes:
        records.append(
            {
                "class_name": class_name,
                "k": k,
                "prompt": template.format(category=class_name),
                "paper_template": template,
                "paper_model": "ChatGPT-4",
                "paper_generation_settings": {
                    "temperature": None,
                    "top_p": None,
                    "max_output_tokens": None,
                    "num_generations": None,
                    "seed_policy": None,
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--template", default=FIG1_TEMPLATE)
    args = parser.parse_args()

    classes = load_classes(args.classes)
    records = build_records(classes, args.k, args.template)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} prompts to {args.out}")


if __name__ == "__main__":
    main()

