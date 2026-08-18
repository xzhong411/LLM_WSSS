#!/usr/bin/env python3
"""Run a one-image LLaVA smoke inference with the local DSV-LFS assets.

"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import torch
from PIL import Image


DEFAULT_DSV_ROOT = Path("/data0/zhongxiang/DSV-LFS-main")
DEFAULT_MODEL_PATH = DEFAULT_DSV_ROOT / "llava-v1.5-7b"
DEFAULT_IMAGE = DEFAULT_DSV_ROOT / "model" / "llava" / "serve" / "examples" / "extreme_ironing.jpg"


def add_llava_paths(dsv_root: Path) -> None:
    for item in [dsv_root, dsv_root / "model"]:
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)


def patch_generate_for_legacy_prepare(model) -> None:
    from llava.constants import IMAGE_TOKEN_INDEX

    def prepare_inference_only(self, input_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        if images is None or input_ids.shape[1] == 1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            return None, attention_mask, past_key_values, inputs_embeds, labels, None

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        new_input_embeds = []
        cur_image_idx = 0
        for cur_input_ids in input_ids:
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if image_token_indices.numel() == 0:
                new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                continue

            cur_new_input_embeds = []
            while image_token_indices.numel() > 0:
                image_token_start = image_token_indices[0]
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start + 1 :]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
            new_input_embeds.append(torch.cat([x.to(self.device) for x in cur_new_input_embeds], dim=0))

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)
            aligned = []
            for cur in new_input_embeds:
                if cur.shape[0] < max_len:
                    pad = torch.zeros(
                        (max_len - cur.shape[0], cur.shape[1]),
                        dtype=cur.dtype,
                        device=cur.device,
                    )
                    cur = torch.cat((cur, pad), dim=0)
                aligned.append(cur)
            inputs_embeds = torch.stack(aligned, dim=0)
        else:
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        return None, attention_mask, past_key_values, inputs_embeds, labels, None

    @torch.no_grad()
    def generate_fixed(self, inputs=None, images=None, image_sizes=None, **kwargs):
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                _,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                attention_mask,
                None,
                None,
                images,
                image_sizes,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super(type(self), self).generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def forward_fixed(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
    ):
        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                _,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
            )

        return super(type(self), self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    model.prepare_inputs_labels_for_multimodal = types.MethodType(prepare_inference_only, model)
    model.forward = types.MethodType(forward_fixed, model)
    model.generate = types.MethodType(generate_fixed, model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsv-root", type=Path, default=DEFAULT_DSV_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-file", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--query", default="Describe this image in one sentence.")
    parser.add_argument("--out", type=Path, default=Path("outputs/llava_smoke/status.json"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    add_llava_paths(args.dsv_root)

    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init

    disable_torch_init()
    model_name = get_model_name_from_path(str(args.model_path))
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        str(args.model_path),
        None,
        model_name,
        device_map="auto",
    )
    patch_generate_for_legacy_prepare(model)

    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(args.image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    status = {
        "ok": True,
        "model_path": str(args.model_path),
        "image_file": str(args.image_file),
        "query": args.query,
        "context_len": context_len,
        "output": output,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
