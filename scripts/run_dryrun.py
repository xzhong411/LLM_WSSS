#!/usr/bin/env python3
"""Run a project-level dry-run.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from llm_eri.mask_rules import resolve_conflicts_max_score
from llm_eri.paper_ops import hypercorrelation, select_top1_confident_part, support_feature_map
from llm_eri.spd import format_category_decomposition_prompt


ROOT = Path(__file__).resolve().parents[1]


def load_classes(path: Path, limit: int) -> list[str]:
    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            classes.append(item)
        if len(classes) >= limit:
            break
    return classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "llm_eri_paper.yaml")
    parser.add_argument("--classes", type=Path, default=ROOT / "data" / "classes" / "voc2012.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "dryrun" / "status.json")
    parser.add_argument("--class-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="Dry-run seed only; the paper does not specify an LLM seed.")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    k = int(config["semantic_part_decomposition"]["reproduction_default_k"])
    class_names = load_classes(args.classes, args.class_limit)

    torch.manual_seed(args.seed)
    h, w, c = 8, 8, 16
    attention_maps = torch.rand(k, h, w)
    attention_maps[2, 3:5, 3:5] += 1.0
    anchor_idx, anchor_scores = select_top1_confident_part(attention_maps, theta=0.6)

    features = torch.randn(c, h, w)
    support = support_feature_map(features, attention_maps[anchor_idx], tau=0.75)
    corr = hypercorrelation(features, support)

    class_scores = torch.sigmoid(torch.randn(len(class_names), h, w))
    label_map = resolve_conflicts_max_score(class_scores, threshold=0.5)

    status = {
        "ok": True,
        "note": "",
        "paper_missing_items_preserved": {
            "gpt_generation_settings": config["semantic_part_decomposition"]["paper_temperature"] is None,
            "llava_layer_for_s_pk": config["model"]["llava_layer_for_s_pk"] is None,
            "paper_pixel_conflict_resolution": config["mask_generation"]["paper_pixel_conflict_resolution"] is None,
        },
        "dryrun_seed": args.seed,
        "default_k": k,
        "classes": class_names,
        "prompts": {name: format_category_decomposition_prompt(name) for name in class_names},
        "anchor": {
            "selected_component_index": anchor_idx,
            "scores": [float(x) for x in anchor_scores.detach().cpu()],
        },
        "shapes": {
            "features": list(features.shape),
            "support": list(support.shape),
            "hypercorrelation": list(corr.shape),
            "label_map": list(label_map.shape),
        },
        "label_values": sorted(int(x) for x in torch.unique(label_map).detach().cpu()),
        "projection_dimension_recorded": config["model"]["visual_token_projection_dimension_paper"],
        "projection_dimension_assumption": config["model"]["visual_token_projection_dimension_reproduction_assumption"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

