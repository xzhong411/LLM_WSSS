



# LLM-Guided Semantic Part Priors and Intra-Image Self-Support for Weakly Supervised Semantic Segmentation



## Quick Start

```bash
cd /data0/zhongxiang/LLM_WSSS
python scripts/generate_spd_prompts.py --classes data/classes/voc2012.txt --out outputs/prompts/voc2012_spd_prompts.jsonl

PYTHONPATH=src /data0/zhongxiang/conda/envs/py39/bin/python scripts/run_dryrun.py
/data0/zhongxiang/conda/envs/gla/bin/python scripts/validate_semantic_parts.py
bash scripts/run_voc_stage1_smoke.sh
```