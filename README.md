# PersonaDual

**Balancing Personalization and Objectivity via Adaptive Reasoning**

Training code for **PersonaDual**: a single LLM that supports both *general-purpose objective reasoning* and *personalized reasoning*, and **adaptively switches** between them given the query and persona.

Two-stage training:

1. **SFT** — learn two reasoning patterns activated by `[General_mode]` / `[Personalized_mode]`
2. **DualGRPO** — RL with forced-prefix sampling and inter-/intra-mode advantages for mode selection

Paper authors: Xiaoyou Liu*, Xinyi Mou* (equal contribution), Shengbin Yue, Liang Wang, Yuqing Wang, Qiexiang Wang, Tianrui Qin, Zhongyu Wei (Fudan University / Shanghai Innovation Institute / OPPO).

## Highlights

- Unified dual-mode reasoning in one model
- DualGRPO reduces mode imbalance during RL and strengthens prefix learning
- Mitigates interference from **unaligned** personas on objective QA
- Leverages **aligned** personas to improve objective accuracy (~+3% on average in the paper)

## Repository Layout

```text
.
├── train-dataprocess/              # Scripts to build PersonaDualData (SFT / RL)
│   ├── sft/
│   └── rl/
├── sft/
│   └── LLaMA-Factory/              # Stage-1 SFT configs & framework
├── rl/
│   ├── rl-data/                    # RL parquet path
│   ├── verl-DualGRPO/              # DualGRPO implementation
│   └── verl/                       # Ablation launch scripts
├── requirements.txt
└── README.md
```

## Method Sketch

Given query \(q\) and persona \(p\), PersonaDual selects a mode and generates with the matching prefix:

| Mode | Prefix | Behavior |
|------|--------|----------|
| General | `[General_mode]` | Downweight / ignore persona; favor factual reasoning |
| Personalized | `[Personalized_mode]` | Explicitly use persona cues |

**DualGRPO** (Stage 2):

- Forced-prefix sampling under both modes for each \((q, p)\)
- Intra-mode advantage (within-mode centering)
- Inter-mode advantage (zero-sum comparison of mode means)
- Prefix strengthening on early mode tokens

## Setup

```bash
git clone <your-repo-url> PersonaDual
cd PersonaDual
pip install -r requirements.txt
```

### Training backends

| Stage | Framework | Path |
|-------|-----------|------|
| SFT | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | `sft/LLaMA-Factory` |
| RL | [verl](https://github.com/volcengine/verl) (modified) | `rl/verl-DualGRPO` |

Install each backend following its own README (CUDA / Ray / vLLM versions matter).

## Data

Sources used in the paper:

- Objective: UltraMedical, FLAN
- Personalized: AlignX
- Personas: PersonaHub (unaligned) + model-generated aligned personas

Construction scripts: `train-dataprocess/`.

Data layout:

| Path | Role |
|------|------|
| `sft/LLaMA-Factory/data/data.json` | Mixed dual-mode SFT (`dualsft`) |
| `sft/LLaMA-Factory/data/data-general.json` | General-mode SFT |
| `sft/LLaMA-Factory/data/data-personalization.json` | Personalized-mode SFT |
| `sft/LLaMA-Factory/data/data-think.json` | CoT / think ablation |
| `rl/rl-data/*.parquet` | RL train / val |

Register SFT files in `sft/LLaMA-Factory/data/dataset_info.json`.

## Training

### Stage 1 — SFT

```bash
cd sft/LLaMA-Factory
# install LLaMA-Factory dependencies, then e.g.:
llamafactory-cli train examples/train_full/llama3_full_sft.yaml
```

Useful configs in `examples/train_full/`:

| Config | Setting |
|--------|---------|
| `llama3_full_sft.yaml` | PersonaDual mixed SFT (`dualsft`) |
| `llama3_full_sft_general.yaml` | General-only ablation |
| `llama3_full_sft_personal.yaml` | Personalized-only ablation |
| `llama3_full_sft_think.yaml` | CoT / think baseline |

Edit `model_name_or_path` and `output_dir` to your local paths.

### Stage 2 — DualGRPO

```bash
cd rl/verl-DualGRPO
# install verl + Ray / vLLM, then:
bash examples/grpo_trainer/run_qwen3-8b.sh
```

Ablation launchers in `rl/verl/examples/grpo_trainer/`:

| Script | Purpose |
|--------|---------|
| `run_qwen3-8b-dual.sh` | Dual-mode RL |
| `run_qwen3-8b-general.sh` | General-only RL |
| `run_qwen3-8b-personal.sh` | Personalized-only RL |

Update in the scripts:

- data train/val files → your parquet paths
- actor model path → Stage-1 SFT checkpoint

Primary backbone in the paper: **Qwen3-8B-Instruct** (non-thinking); Llama-3.1-8B results in the appendix.

DualGRPO changes vs upstream verl: forced `[General_mode]` / `[Personalized_mode]` prefixes in rollout, dual-mode advantage decomposition in the trainer, and mode-aware reward formatting.


## Acknowledgments

Built with:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — SFT
- [verl](https://github.com/volcengine/verl) — RL / GRPO
- [AlignX](https://github.com/JinaLeejnl/AlignX) — personalization data and baselines
