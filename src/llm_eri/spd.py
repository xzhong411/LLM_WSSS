"""Semantic Part Decomposition helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass


FIG1_PROMPT_TEMPLATE = "Decompose category [{category}] into semantic parts decoupling"


@dataclass(frozen=True)
class SemanticPart:
    name: str
    description: str | None = None


def format_category_decomposition_prompt(category: str) -> str:
    return FIG1_PROMPT_TEMPLATE.format(category=category.strip())


def parse_part_description(text: str) -> SemanticPart:
  
    clean = text.strip().strip("-*0123456789. ")
    pieces = re.split(r"\s[-:]\s", clean, maxsplit=1)
    if len(pieces) == 1:
        return SemanticPart(name=pieces[0].strip(), description=None)
    return SemanticPart(name=pieces[0].strip(), description=pieces[1].strip())


def normalize_part_name_for_cache(name: str) -> str:

    return re.sub(r"\s+", " ", name.strip().lower())

