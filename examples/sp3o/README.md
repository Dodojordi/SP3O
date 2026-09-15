# SP3O reproduction guide

SP3O changes only the token positions used to reduce the PPO critic value loss.
The actor loss, rollout targets, and PPO advantage calculation remain unchanged.
This implementation is based on slime `v0.2.4` (`bd217a63`).

## 1. Environment and data

Install slime and Megatron-LM by following the upstream
[quick-start guide](../../docs/en/get_started/quick_start.md). The three runs
use Qwen3-4B-Base on eight GPUs: four colocated actor/critic GPUs and four
rollout GPUs. Resource parameters can be overridden for another topology.

Training data must be JSONL. Each row contains a chat-formatted `prompt`, a
`label` string or list of strings, and optionally `metadata.points` for
multi-answer problems:

```json
{"prompt":[{"role":"user","content":"..."}],"label":["42"]}
```

Copy `.env.example` to a file outside Git, set the five required paths, and
export them. Do not place API keys or machine-specific paths in tracked files.

## 2. Validate a configuration

`DRY_RUN=1` validates the required settings and experiment name, then
prints the resolved training command without starting Ray or using GPUs:

```bash
set -a
source /path/to/private.env
set +a
DRY_RUN=1 bash examples/sp3o/train_sp3o.sh
```

The local `sp3o_math` reward reproduces the rule-based boxed-answer path used
for training. It never calls an external judge or sends prompts over the
network.

## 3. Main runs

```bash
bash examples/sp3o/train_ppo.sh
bash examples/sp3o/train_grpo.sh
bash examples/sp3o/train_sp3o.sh
```

The `sp3o` preset supervises relative response positions `0.3/0.6/0.9` and
adds `0.95` when a response has at least 6144 valid tokens. PPO critic warm-up
is dense for the first 20 rollout batches. Set `NUM_CRITIC_ONLY_STEPS` only
when intentionally changing the warm-up duration, such as for a smoke test.

## 4. Evaluation and resuming

Set `EVAL_CONFIG` to a slime evaluation YAML such as `eval.example.yaml`.
Replace every placeholder path before use. Per-dataset sample counts belong in
the YAML so the 32-sample mathematical evaluation can be reproduced without
changing the launcher.

The launcher writes actor, critic, W&B, and logs beneath
`$OUTPUT_DIR/$EXPERIMENT_NAME`. For a resumed run, point `MEGATRON_CHECKPOINT`
to the actor checkpoint and pass the corresponding standard slime resume
arguments by extending the launcher locally; checkpoint rewriting is kept out
of this reference implementation.

## Public implementation flags

- `--critic-token-loss`: enable sparse critic supervision.
- `--critic-token-ratios`: sorted, unique response-relative anchors in `[0,1]`.
- `--critic-extra-tail-ratio`: optional conditional late anchor.
- `--critic-extra-tail-min-response-len`: valid-token threshold for that anchor.
- `--dense-critic-only-warmup`: use dense critic masks during critic-only warm-up.

Invalid ratios, missing selectors, or inconsistent warm-up settings fail before
training begins.
