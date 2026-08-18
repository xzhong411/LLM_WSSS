#!/usr/bin/env python3
import os
import sys

import torch


DSV_ROOT = os.environ.get("DSV_ROOT", "/data0/zhongxiang/DSV-LFS-main")
if DSV_ROOT not in sys.path:
    sys.path.insert(0, DSV_ROOT)

from model.DSVLFS import DSVLFSForCausalLM  # noqa: E402
from model.DSVLFS_WeaklySupervised import WeaklySupervisedDSVLFS  # noqa: E402


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _finite(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)


def _get_evolved_semantic_prompts_compat(self, images_clip, input_ids, attention_masks):
    batch_size = images_clip.shape[0]

    with torch.no_grad():
        output, seg_token_vec_align = super(DSVLFSForCausalLM, self).forward(
            images=images_clip,
            attention_mask=attention_masks,
            input_ids=input_ids,
            output_hidden_states=True,
        )
        hidden_states = output.hidden_states[-1].detach()

    sem_token_mask = seg_token_vec_align == self.sem_token_idx
    sem_embeddings_list = []
    for i in range(batch_size):
        if sem_token_mask[i].any():
            sem_idx = torch.where(sem_token_mask[i])[0][0]
            sem_embedding = hidden_states[i, sem_idx]
        else:
            sem_embedding = hidden_states[i, -1]
        sem_embeddings_list.append(sem_embedding)

    sem_embeddings = _finite(torch.stack(sem_embeddings_list, dim=0))

    shape_feats = _finite(self.shape_projector(
        sem_embeddings.to(_module_dtype(self.shape_projector))
    ).float())
    texture_feats = _finite(self.texture_projector(
        sem_embeddings.to(_module_dtype(self.texture_projector))
    ).float())
    spatial_feats = _finite(self.spatial_projector(
        sem_embeddings.to(_module_dtype(self.spatial_projector))
    ).float())

    return shape_feats, texture_feats, spatial_feats


def _mask_decoder_with_attention_compat(self, image_embeddings, semantic_prompts, visual_prompt):
    proj_dtype = _module_dtype(self.query_proj)
    image_embeddings = _finite(image_embeddings).to(proj_dtype)
    visual_prompt = _finite(visual_prompt).to(proj_dtype)
    shape_feats, texture_feats, spatial_feats = [
        _finite(item).to(proj_dtype) for item in semantic_prompts
    ]

    batch_size, _, height, width = image_embeddings.shape
    semantic_fused = (shape_feats + texture_feats + spatial_feats).unsqueeze(-1).unsqueeze(-1)
    semantic_expanded = semantic_fused.expand(-1, -1, height, width)

    query = self.query_proj(image_embeddings.flatten(2).permute(2, 0, 1))
    key = self.key_proj(semantic_expanded.flatten(2).permute(2, 0, 1))
    value = self.value_proj(semantic_expanded.flatten(2).permute(2, 0, 1))

    attn_output, _ = self.attention(query, key, value)
    attn_output = attn_output.permute(1, 2, 0).reshape(batch_size, 256, height, width)

    mask_dtype = _module_dtype(self.mask_head)
    fused_features = torch.cat([attn_output, image_embeddings, visual_prompt], dim=1)
    return _finite(self.mask_head(fused_features.to(mask_dtype)).float())


_original_dense_matching_module = WeaklySupervisedDSVLFS.dense_matching_module
_original_generate_pseudo_labels = WeaklySupervisedDSVLFS.generate_pseudo_labels
_original_refine_with_sam = WeaklySupervisedDSVLFS.refine_with_sam


def _dense_matching_module_compat(self, image_features, evolved_semantic_prompts):
    prompts = tuple(_finite(item) for item in evolved_semantic_prompts)
    return _finite(_original_dense_matching_module(self, _finite(image_features), prompts))


def _generate_pseudo_labels_compat(self, image_features, evolved_semantic_prompts):
    pseudo_labels, pseudo_heatmap, uncertainty = _original_generate_pseudo_labels(
        self,
        _finite(image_features),
        tuple(_finite(item) for item in evolved_semantic_prompts),
    )
    return _finite(pseudo_labels), _finite(pseudo_heatmap), _finite(uncertainty)


def _refine_with_sam_compat(self, image_embeddings, pseudo_heatmap, uncertainty, resize_list):
    return _finite(
        _original_refine_with_sam(
            self,
            _finite(image_embeddings),
            _finite(pseudo_heatmap),
            _finite(uncertainty),
            resize_list,
        )
    )


WeaklySupervisedDSVLFS.get_evolved_semantic_prompts = _get_evolved_semantic_prompts_compat
WeaklySupervisedDSVLFS.mask_decoder_with_attention = _mask_decoder_with_attention_compat
WeaklySupervisedDSVLFS.dense_matching_module = _dense_matching_module_compat
WeaklySupervisedDSVLFS.generate_pseudo_labels = _generate_pseudo_labels_compat
WeaklySupervisedDSVLFS.refine_with_sam = _refine_with_sam_compat

import train_weakly as _train_weakly  # noqa: E402


_original_create_model_tokenizer = _train_weakly.CreateModelTokenizer


def _create_model_tokenizer_paper_freeze(args):
    model, tokenizer = _original_create_model_tokenizer(args)

    trainable_prefixes = (
        "model.visual_model.mask_decoder.",
        "shape_projector.",
        "texture_projector.",
        "spatial_projector.",
        "feature_projector.",
        "hypercorr_encoder.",
        "hypercorr_decoder.",
        "pseudo_label_generator.",
        "uncertainty_estimator.",
        "query_proj.",
        "key_proj.",
        "value_proj.",
        "attention.",
        "mask_head.",
    )

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)

    if getattr(args, "local_rank", 0) == 0:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(
            f"[LLM-ERI compat] trainable params: {trainable:,} / {total:,} "
            f"({trainable / total:.4%})"
        )
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("[LLM-ERI compat] trainable:", name, tuple(param.shape))

    return model, tokenizer


_train_weakly.CreateModelTokenizer = _create_model_tokenizer_paper_freeze


if __name__ == "__main__":
    _train_weakly.main(sys.argv[1:])
