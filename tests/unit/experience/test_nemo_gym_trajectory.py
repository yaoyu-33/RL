import pytest
import torch

from nemo_rl.environments.nemo_gym_trajectory import trajectory_to_completion
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.payload import record_to_train_batch


def test_gym_trajectory_reaches_nemo_rl_training_batch():
    trajectory = {
        "task_id": "arithmetic",
        "sample_id": "arithmetic:0",
        "input_ids": [10, 11, 12, 13],
        "loss_mask": [0, 0, 1, 1],
        "logprobs": [0.0, 0.0, -0.1, -0.2],
        "reward": 1.0,
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
    }

    completion = trajectory_to_completion(trajectory)
    record = PromptGroupRecord(
        prompt_idx=0,
        prompt=[completion.message_log[0]],
        extra_env_info=None,
        metadata={"task_name": "arithmetic"},
        completions=[completion],
        rollout_metrics={},
    )
    batch = record_to_train_batch(record, pad_value_dict={"token_ids": 0})

    assert batch["input_ids"].tolist() == [[10, 11, 12, 13]]
    assert batch["token_mask"].tolist() == [[0, 0, 1, 1]]
    assert torch.allclose(
        batch["generation_logprobs"], torch.tensor([[0.0, 0.0, -0.1, -0.2]])
    )
    assert batch["total_reward"].tolist() == [1.0]


@pytest.mark.parametrize("field", ["input_ids", "loss_mask", "logprobs", "reward"])
def test_adapter_rejects_missing_training_field(field):
    trajectory = {
        "input_ids": [10, 11],
        "loss_mask": [0, 1],
        "logprobs": [0.0, -0.1],
        "reward": 1.0,
    }
    trajectory.pop(field)

    with pytest.raises(ValueError):
        trajectory_to_completion(trajectory)
