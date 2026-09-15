#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PRESET="${PRESET:-sp3o}"
SEED="${SEED:-1234}"
DRY_RUN="${DRY_RUN:-0}"

required=(HF_CHECKPOINT MEGATRON_CHECKPOINT PROMPT_DATA OUTPUT_DIR MEGATRON_LM)
for name in "${required[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
done

case "${PRESET}" in
    ppo|grpo|sp3o) ;;
    *) echo "Unknown PRESET: ${PRESET}" >&2; exit 2 ;;
esac

# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/models/qwen3-4B.sh"
MODEL_ARGS+=(--max-position-embeddings 32768 --seq-length 32768)

CRITIC_RATIOS=(0.3 0.6 0.9)

ACTOR_GPUS="${ACTOR_GPUS:-4}"
TOTAL_GPUS="${TOTAL_GPUS:-8}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-8192}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-2}"
NUM_ROLLOUT="${NUM_ROLLOUT:-1000}"
NUM_CRITIC_ONLY_STEPS="${NUM_CRITIC_ONLY_STEPS:-20}"
OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-$((2 * ROLLOUT_BATCH_SIZE))}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen3-4B-Base-${PRESET}-seed${SEED}}"
RUN_DIR="${OUTPUT_DIR%/}/${EXPERIMENT_NAME}"
if [[ ! "${NUM_CRITIC_ONLY_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "NUM_CRITIC_ONLY_STEPS must be a non-negative integer" >&2
    exit 2
fi
CUDA_GRAPH_BATCH_SIZES=(1 2 4 8)
for ((batch_size = 16; batch_size <= 256; batch_size += 8)); do
    CUDA_GRAPH_BATCH_SIZES+=("${batch_size}")
done

ARGS=(
    "${MODEL_ARGS[@]}"
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${ACTOR_GPUS}"
    --colocate
    --megatron-to-hf-mode bridge
    --hf-checkpoint "${HF_CHECKPOINT}"
    --ref-load "${MEGATRON_CHECKPOINT}"
    --load "${MEGATRON_CHECKPOINT}"
    --save "${RUN_DIR}/actor"
    --save-interval "${SAVE_INTERVAL}"
    --prompt-data "${PROMPT_DATA}"
    --input-key "${INPUT_KEY:-prompt}"
    --label-key "${LABEL_KEY:-label}"
    --apply-chat-template
    --rollout-shuffle
    --rm-type sp3o_math
    --reward-key score
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --partial-rollout
    --balance-data
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
    --tensor-model-parallel-size "${ACTOR_GPUS}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --transformer-impl transformer_engine
    --bf16
    --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
    --sglang-cuda-graph-bs "${CUDA_GRAPH_BATCH_SIZES[@]}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --use-tis
    --seed "${SEED}"
)

if [[ "${PRESET}" == "grpo" ]]; then
    ARGS+=(
        --advantage-estimator grpo
        --kl-loss-coef 0.0
        --kl-loss-type low_var_kl
        --kl-coef 0.0
        --eps-clip 0.2
        --eps-clip-high 0.2
    )
else
    ARGS+=(
        --advantage-estimator ppo
        --kl-loss-coef 0.0
        --kl-loss-type low_var_kl
        --kl-coef 0.0
        --eps-clip 0.2
        --eps-clip-high 0.2
        --lambd 1.0
        --num-critic-only-steps "${NUM_CRITIC_ONLY_STEPS}"
        --critic-lr 4e-6
        --critic-num-nodes 1
        --critic-num-gpus-per-node "${ACTOR_GPUS}"
        --critic-save "${RUN_DIR}/critic"
    )
fi

if [[ "${PRESET}" == "sp3o" ]]; then
    ARGS+=(--critic-token-loss --critic-token-ratios "${CRITIC_RATIOS[@]}")
    if ((NUM_CRITIC_ONLY_STEPS > 0)); then
        ARGS+=(--dense-critic-only-warmup)
    fi
    ARGS+=(--critic-extra-tail-ratio 0.95 --critic-extra-tail-min-response-len 6144)
fi

if [[ -n "${EVAL_CONFIG:-}" ]]; then
    ARGS+=(
        --eval-config "${EVAL_CONFIG}"
        --eval-interval "${EVAL_INTERVAL:-10}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-24576}"
        --eval-top-p 0.95
        --eval-temperature 1.0
        --log-passrate
    )
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
    ARGS+=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT:-SP3O}"
        --wandb-group "${EXPERIMENT_NAME}"
        --disable-wandb-random-suffix
        --wandb-dir "${RUN_DIR}/wandb"
        --wandb-mode "${WANDB_MODE:-online}"
    )
fi

DISPLAY_ANCHORS="dense"
if [[ "${PRESET}" == "sp3o" ]]; then
    DISPLAY_ANCHORS="${CRITIC_RATIOS[*]}"
fi
printf 'SP3O configuration: model=%s preset=%s seed=%s anchors=%s\n' \
    "Qwen3-4B-Base" "${PRESET}" "${SEED}" "${DISPLAY_ANCHORS}"
printf 'Command:'
printf ' %q' python3 train.py "${ARGS[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

mkdir -p "${RUN_DIR}"
export PYTHONBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

if [[ "${START_RAY:-1}" == "1" ]]; then
    ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${TOTAL_GPUS}" \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port "${RAY_DASHBOARD_PORT:-26500}"
fi

RUNTIME_ENV_JSON="$(python3 -c 'import json, os; env={"PYTHONPATH": os.environ["MEGATRON_LM"], "CUDA_DEVICE_MAX_CONNECTIONS": "1"}; key=os.environ.get("WANDB_API_KEY"); env.update({"WANDB_API_KEY": key} if key else {}); print(json.dumps({"env_vars": env}))')"
cd "${REPO_ROOT}"
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT:-26500}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" -- python3 train.py "${ARGS[@]}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"
