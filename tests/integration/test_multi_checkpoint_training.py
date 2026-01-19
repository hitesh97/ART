"""Integration tests for multi-checkpoint inference with real training loops.

These tests run actual training loops with different backends to verify that
multi-checkpoint inference works end-to-end without crashing.

Usage:
    # Run all integration tests (requires appropriate backend setup)
    uv run pytest tests/integration/test_multi_checkpoint_training.py -v -s

Environment variables:
    BASE_MODEL: The base model to use (default: Qwen/Qwen3-0.6B)
    WANDB_API_KEY: Required for ServerlessBackend test
    TINKER_API_KEY: Required for TinkerBackend test
"""

import os
import tempfile
import uuid

import openai
import pytest

import art
from art.local import LocalBackend

# Use a small model for fast testing
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B"


def get_base_model() -> str:
    """Get the base model to use for testing."""
    return os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL)


async def simple_rollout(
    client: openai.AsyncOpenAI, model_name: str, prompt: str
) -> art.Trajectory:
    """A simple rollout function for testing."""
    messages: art.Messages = [{"role": "user", "content": prompt}]
    chat_completion = await client.chat.completions.create(
        messages=messages,
        model=model_name,
        max_tokens=10,
        timeout=60,
        temperature=1,
    )
    choice = chat_completion.choices[0]
    content = (choice.message.content or "").lower()
    if "yes" in content:
        reward = 1.0
    elif "no" in content:
        reward = 0.5
    elif "maybe" in content:
        reward = 0.25
    else:
        reward = 0.0
    return art.Trajectory(messages_and_choices=[*messages, choice], reward=reward)


async def run_training_loop(
    model: art.TrainableModel,
    num_steps: int = 1,
    rollouts_per_step: int = 4,
) -> list[int]:
    """Run a simple training loop and return the step numbers after each train call."""
    openai_client = model.openai_client()
    prompts = ["Say yes", "Say no", "Say maybe", "Say hello"]
    steps_completed = []

    async def resolve_model_name(preferred: str, fallback: str) -> str:
        try:
            available = [m.id async for m in openai_client.models.list()]
        except Exception:
            return preferred
        return preferred if preferred in available else fallback

    for _ in range(num_steps):
        current_step = await model.get_step()
        preferred_name = model.get_inference_name(step=current_step)
        model_name = await resolve_model_name(
            preferred_name, model.get_inference_name(step=0)
        )
        train_groups = await art.gather_trajectory_groups(
            [
                art.TrajectoryGroup(
                    [
                        simple_rollout(openai_client, model_name, prompt)
                        for _ in range(rollouts_per_step)
                    ]
                )
                for prompt in prompts
            ]
        )
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )
        steps_completed.append(await model.get_step())

    return steps_completed


async def _run_inference_on_step(
    model: art.TrainableModel,
    step: int,
) -> None:
    openai_client = model.openai_client()
    model_name = model.get_inference_name(step=step)
    await openai_client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello"}],
        model=model_name,
        max_tokens=10,
        timeout=30,
    )


@pytest.mark.skipif(
    "TINKER_API_KEY" not in os.environ,
    reason="TINKER_API_KEY not set - skipping TinkerBackend test",
)
async def test_tinker_backend():
    """Test multi-checkpoint inference with TinkerBackend."""
    model_name = f"test-multi-ckpt-tinker-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = art.TinkerBackend(path=tmpdir)
        model = art.TrainableModel(
            name=model_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        try:
            await model.register(backend)
            steps = await run_training_loop(model, num_steps=1, rollouts_per_step=2)
            await _run_inference_on_step(model, step=steps[-1])
            await _run_inference_on_step(model, step=0)
        finally:
            await backend.close()


@pytest.mark.skipif(
    not os.path.exists("/dev/nvidia0"),
    reason="No GPU available - skipping LocalBackend test",
)
async def test_local_backend():
    """Test multi-checkpoint inference with LocalBackend (UnslothService)."""
    model_name = f"test-multi-ckpt-local-{uuid.uuid4().hex[:8]}"
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(path=tmpdir)
        model = art.TrainableModel(
            name=model_name,
            project="integration-tests",
            base_model=get_base_model(),
        )
        try:
            await model.register(backend)
            steps = await run_training_loop(model, num_steps=1, rollouts_per_step=2)
            await _run_inference_on_step(model, step=steps[-1])
            await _run_inference_on_step(model, step=0)
        finally:
            await backend.close()


@pytest.mark.skipif(
    "WANDB_API_KEY" not in os.environ,
    reason="WANDB_API_KEY not set - skipping ServerlessBackend test",
)
async def test_serverless_backend():
    """Test multi-checkpoint inference with ServerlessBackend."""
    model_name = f"test-multi-ckpt-serverless-{uuid.uuid4().hex[:8]}"
    backend = art.ServerlessBackend()
    model = art.TrainableModel(
        name=model_name,
        project="integration-tests",
        base_model="meta-llama/Llama-3.1-8B-Instruct",
    )
    try:
        await model.register(backend)
        steps = await run_training_loop(model, num_steps=1, rollouts_per_step=2)
        await _run_inference_on_step(model, step=steps[-1])
        await _run_inference_on_step(model, step=0)
    finally:
        try:
            await backend.delete(model)
        except Exception:
            pass
        await backend.close()
