# SP3O

Official repository for the paper **"Rethinking Critic Learning in PPO:
Understanding and Mitigating Value Flattening."**

[[Project Page](https://dodojordi.github.io/SP3O/)] [[Paper](https://arxiv.org/abs/2609.18708)] [[HF Paper](https://huggingface.co/papers/2609.18708)]

This repository is built on
[THUDM/slime v0.2.4](https://github.com/THUDM/slime/tree/v0.2.4).

## Quick Start

Configure the required paths:

```bash
cp examples/sp3o/.env.example .env
# Edit .env with your checkpoint, data, Megatron-LM, and output paths.
set -a
source .env
set +a
```

Launch a Qwen3-4B-Base experiment with eight GPUs:

```bash
bash examples/sp3o/train_sp3o.sh
bash examples/sp3o/train_ppo.sh
bash examples/sp3o/train_grpo.sh
```
