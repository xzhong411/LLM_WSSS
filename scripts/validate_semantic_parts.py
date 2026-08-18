#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=ROOT / "data" / "semantic_parts" / "voc2012_codex_k5.json")
    args = parser.parse_args()

    data = json.loads(args.file.read_text(encoding="utf-8"))
    expected_k = int(data["metadata"]["k"])
    classes = data["classes"]
    errors = []
    for class_name, parts in classes.items():
        if len(parts) != expected_k:
            errors.append(f"{class_name}: expected {expected_k}, got {len(parts)}")
        names = [p["name"].strip().lower() for p in parts]
        if len(names) != len(set(names)):
            errors.append(f"{class_name}: duplicate part names")
        for part in parts:
            if not part.get("name") or not part.get("description"):
                errors.append(f"{class_name}: empty part or description")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(classes)} classes with K={expected_k}: {args.file}")


if __name__ == "__main__":
    main()

