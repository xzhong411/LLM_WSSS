#!/usr/bin/env python3
"""Probe local LLaVA code and weights"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


DEFAULT_DSV_ROOT = Path("/data0/zhongxiang/DSV-LFS-main")
DEFAULT_LLAVA_MODEL = DEFAULT_DSV_ROOT / "llava-v1.5-7b"


def add_llava_paths(dsv_root: Path) -> None:
    for item in [dsv_root, dsv_root / "model"]:
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsv-root", type=Path, default=DEFAULT_DSV_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LLAVA_MODEL)
    parser.add_argument("--out", type=Path, default=Path("outputs/llava_probe/status.json"))
    args = parser.parse_args()

    add_llava_paths(args.dsv_root)

    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from llava.conversation import conv_templates
    from llava.constants import DEFAULT_IMAGE_TOKEN
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model  # noqa: F401

    config = AutoConfig.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nDescribe the semantic parts of the target object.")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    files = {}
    for name in [
        "config.json",
        "tokenizer.model",
        "mm_projector.bin",
        "pytorch_model.bin.index.json",
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
    ]:
        path = args.model_path / name
        files[name] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else None}

    status = {
        "ok": True,
        "host": os.uname().nodename,
        "dsv_root": str(args.dsv_root),
        "llava_code": str(args.dsv_root / "model" / "llava"),
        "model_path": str(args.model_path),
        "model_name": get_model_name_from_path(str(args.model_path)),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "config_class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vision_tower": getattr(config, "mm_vision_tower", None),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": len(tokenizer),
        "files": files,
        "conversation_mode": "llava_v1",
        "example_input_format": prompt,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

