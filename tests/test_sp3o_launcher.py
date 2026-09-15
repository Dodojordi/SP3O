import os
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).parents[1]
LAUNCHER = REPO_ROOT / "examples" / "sp3o" / "train.sh"
ALL_PRESETS = ["ppo", "grpo", "sp3o"]
PUBLIC_LAUNCHERS = {
    "ppo": REPO_ROOT / "examples" / "sp3o" / "train_ppo.sh",
    "grpo": REPO_ROOT / "examples" / "sp3o" / "train_grpo.sh",
    "sp3o": REPO_ROOT / "examples" / "sp3o" / "train_sp3o.sh",
}


def _dry_run(preset: str, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PRESET": preset,
            "HF_CHECKPOINT": "/public/hf",
            "MEGATRON_CHECKPOINT": "/public/megatron",
            "PROMPT_DATA": "/public/train.jsonl",
            "OUTPUT_DIR": "/public/output",
            "MEGATRON_LM": "/public/Megatron-LM",
            "WANDB_API_KEY": "launcher-test-secret",
        }
    )
    env.update(overrides)
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_every_public_preset_has_a_valid_dry_run(preset):
    result = _dry_run(preset)

    assert result.returncode == 0, result.stderr
    assert "launcher-test-secret" not in result.stdout
    assert "--use-tis" in result.stdout
    assert "--partial-rollout" in result.stdout
    assert "--over-sampling-batch-size 128" in result.stdout
    assert "--optimizer-cpu-offload" in result.stdout
    assert "--save-interval 10" in result.stdout
    assert "--sglang-cuda-graph-bs 1 2 4 8 16" in result.stdout


def test_sp3o_preset_resolves_paper_configuration():
    result = _dry_run("sp3o")

    assert "anchors=0.3 0.6 0.9" in result.stdout
    assert "--critic-token-ratios 0.3 0.6 0.9" in result.stdout
    assert "--critic-extra-tail-ratio 0.95" in result.stdout
    assert "--critic-extra-tail-min-response-len 6144" in result.stdout
    assert "--dense-critic-only-warmup" in result.stdout


def test_zero_warmup_smoke_run_immediately_uses_sparse_critic():
    result = _dry_run("sp3o", NUM_CRITIC_ONLY_STEPS="0", NUM_ROLLOUT="1")

    assert result.returncode == 0, result.stderr
    assert "--num-critic-only-steps 0" in result.stdout
    assert "--critic-token-loss" in result.stdout
    assert "--dense-critic-only-warmup" not in result.stdout


@pytest.mark.parametrize("preset", ALL_PRESETS)
def test_public_launcher_selects_its_named_experiment(preset):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "PRESET": "must-be-overridden",
            "HF_CHECKPOINT": "/public/hf",
            "MEGATRON_CHECKPOINT": "/public/megatron",
            "PROMPT_DATA": "/public/train.jsonl",
            "OUTPUT_DIR": "/public/output",
            "MEGATRON_LM": "/public/Megatron-LM",
            "WANDB_API_KEY": "",
        }
    )

    result = subprocess.run(
        ["bash", str(PUBLIC_LAUNCHERS[preset])],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"preset={preset}" in result.stdout


def test_ppo_and_grpo_do_not_enable_sparse_critic_loss():
    ppo = _dry_run("ppo").stdout
    grpo = _dry_run("grpo").stdout

    assert "--critic-token-loss" not in ppo
    assert "--critic-token-loss" not in grpo
    assert "--advantage-estimator ppo" in ppo
    assert "--advantage-estimator grpo" in grpo
    assert "--critic-save" not in grpo


def test_unknown_preset_fails_before_launch():
    result = _dry_run("unknown")

    assert result.returncode == 2
    assert "Unknown PRESET" in result.stderr


def test_invalid_warmup_duration_fails_before_launch():
    result = _dry_run("sp3o", NUM_CRITIC_ONLY_STEPS="-1")

    assert result.returncode == 2
    assert "NUM_CRITIC_ONLY_STEPS must be a non-negative integer" in result.stderr
