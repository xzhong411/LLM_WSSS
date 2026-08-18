# Paper Parameter Audit

Source: `paper/TMM_four (2).pdf`

The table below answers the 12 requested reproduction details. "Not specified" means the PDF does not provide the exact value, even if a reasonable implementation default can be inferred from common LLaVA/SAM practice.

| # | Requested detail | Paper value | Reproduction note |
|---|---|---|---|
| 1 | Exact category-decomposition prompt template | `Decompose category [{category}] into semantic parts decoupling` | This is the exact prompt visible in Fig. 1 for the example `bird`. No longer system/user template is provided. |
| 2 | GPT model/version used | `ChatGPT-4`; Fig. 1 also labels `GPT-4` | No API model ID or dated snapshot is specified. |
| 3 | Temperature, top-p, max output length, number of generations | Not specified | Fig. 6(a) says four prompt variations were tested, but it does not list generation sampling settings. |
| 4 | Same random seed for repeated generations | Not specified | No LLM seed, decoding seed, or repeated-generation protocol is given. |
| 5 | Default number of semantic components K | The PDF marks `Optimal K=5` in Fig. 6(b) | The scaffold uses K=5 as the default; the text also says K in [4, 6] is a stable plateau. |
| 6 | Normalization, deduplication, filtering rules for generated descriptions | Not specified | The paper shows part-description pairs such as `Beak - ...` and `Wings - ...`, but gives no post-processing rules. |
| 7 | Input format used by the multimodal model | `S_pk = F_LLM(T(P_C) | Proj(F))` | The text says dense visual tokens `Proj(F)` are fed with textual component descriptor `T(P_C)`. Exact LLaVA conversation/image-token format is not specified. |
| 8 | Visual-token projection dimension | "match the LLM's input dimension" | Numeric dimension is not in the PDF. This scaffold records 4096 only as a standard LLaVA-v1.5-7b assumption, not a paper-confirmed value. |
| 9 | LLaVA layer and token representation used to construct S_pk | Not specified | The paper names the output `S_pk` from `F_LLM` but does not state which hidden layer or token pooling/selection is used. |
| 10 | Frozen/trainable states of backbone, multimodal model, 4D module, mask decoder | Backbone: frozen; multimodal LLM: frozen; 4D encoder/decoder: trainable; mask decoder: trainable | This comes from Fig. 1 frozen/trainable icons. Projection layers and MLP head are not explicitly labeled; scaffold treats them as trainable assumptions. |
| 11 | Number of multimodal forward passes per image | Not specified | Fig. 1 visually suggests one multimodal pass can output `{S_p1, ..., S_pk}` for one image-level class, while Eq. (1) is written per component. Multi-label images are not resolved in the PDF. |
| 12 | Multi-label class-wise mask generation and pixel-level conflict-resolution rule | Not specified | The paper gives a per-query mask equation `M_q = sigmoid(MLP(Concat(A_1, ..., A_K)))` but no multi-class merge, argmax, thresholding, or overlap rule. |

## Other Paper-Specified Values

| Item | Value |
|---|---|
| Datasets | PASCAL VOC 2012, MS COCO 2014 |
| VOC split | 10,582 augmented train, 1,449 val, 1,456 test |
| COCO split | about 80,000 train, 40,000 val |
| Backbones | ResNet-101, CLIP-ViT |
| Multimodal model | LLaVA-v1.5-7b + Segment Anything |
| Mask decoder | Derived from Segment Anything mask decoder |
| 4D module | 3x3x3x3 4D conv, GroupNorm, ReLU, symmetric encoder-decoder, AvgPool |
| Anchor strategy | Top-1 confident part; top-3 average and random intra-object are ablations |
| Training epochs | 20 |
| Optimizer | AdamW |
| Initial learning rate | 2e-5 |
| Batch size | 2 |
| Loss weights | lambda_1=0.5, lambda_2=0.1, lambda_3=0.5 |
| Hardware | NVIDIA A100 GPUs |
| Reported pseudo-label mIoU | 80.7 on PASCAL VOC |
| Reported final mIoU | VOC val 77.9, VOC test 77.5, COCO val 49.7 |

## Reproduction Blockers

- The exact ChatGPT/GPT-4 endpoint or model snapshot is missing.
- LLM decoding parameters are missing.
- Semantic part post-processing is missing.
- LLaVA feature extraction layer and token representation are missing.
- Numeric visual-token projection dimension is missing from the paper.
- Multi-label image handling and overlapping-mask conflict resolution are missing.
- Dataset preprocessing, crop/resize, augmentation, evaluation scripts, and pseudo-label thresholds are not specified.

