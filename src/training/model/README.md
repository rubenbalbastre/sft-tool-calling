---
base_model: google/gemma-4-E2B-it
library_name: peft
pipeline_tag: text-generation
datasets:
  - rubenbalbastre/supply-chain-tool-calling
language:
  - en
  - es
  - de
  - fr
tags:
  - base_model:adapter:google/gemma-4-E2B-it
  - tool-calling
  - function-calling
  - procurement
  - lora
  - sft
  - transformers
  - trl
---

# Procurement Function Calling — Gemma 4 E2B LoRA SFT

This repository contains a LoRA adapter for `google/gemma-4-E2B-it`, supervised
fine-tuned for multilingual, multi-turn procurement tool calling. The model
searches suppliers, requests quotes, evaluates delivery options and submits a
procurement plan while respecting budget, delivery, reliability and compliance
constraints.

## Results

Every run evaluates **all 400 scenario–prompt pairs** in the held-out test
split, including **unseen templates 16–25**. All configurations use the same
seed and test examples. Values are the mean ± sample standard deviation across
three repeated inference runs.

| Model | Thinking | Runs | Training tokens | Task success | Average return | Mean episode latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | Disabled | 3 | — | 16.58% ± 0.29 pp | 0.230 ± 0.010 | 15.75 ± 0.14 s |
| Base | Enabled | 3 | — | 40.33% ± 1.66 pp | 0.476 ± 0.017 | 95.36 ± 2.04 s |
| LoRA SFT (48 steps) | Disabled | 3 | **1.47M** | **42.50% ± 0.35 pp** | **0.567 ± 0.005** | **20.31 ± 0.19 s** |

Against the non-thinking base, SFT improves mean task success by **25.92
percentage points** and mean return by **0.338**. It also slightly exceeds the
thinking-enabled base while using **about one fifth of its mean episode
latency**.

The 48-step job processed **1,472,140 non-padding input tokens** across 384
examples. Of these, 188,297 assistant tool-call tokens carried loss. Training
took **65 minutes on one NVIDIA A40 GPU**. The run was deliberately
compute-bounded and was not optimized for maximum environment return or task
success.

## Loading

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_id = "rubenbalbastre/procurement-function-calling-gemma-4-E2B-it-sft"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
```

This repository contains the adapter rather than the complete base-model
weights. Use the tokenizer's chat template and provide the procurement tool
schemas when constructing requests.

## Training

- Base model: [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it)
- Dataset: [`rubenbalbastre/supply-chain-tool-calling`](https://huggingface.co/datasets/rubenbalbastre/supply-chain-tool-calling)
- Method: assistant-only LoRA supervised fine-tuning with PEFT and TRL
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Optimizer steps: 48
- Thinking during SFT: disabled

## Limitations

- Results are specific to the included synthetic procurement environment,
  tools and verifier.
- Preferred-supplier fallback remains the weakest task family.
- The model can emit invalid calls, select infeasible plans or fail to complete
  longer trajectories.
- Do not use its output as an autonomous purchasing decision without external
  validation.

## Source

Training, data-generation and evaluation code is available in the
[`sft-tool-calling`](https://github.com/rubenbalbastre/sft-tool-calling)
repository.
