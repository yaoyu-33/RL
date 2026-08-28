import pytest
import torch

from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.environments.nemo_gym_trajectory import trajectory_to_completion


def test_gym_trajectory_reaches_nemo_rl_flat_training_tensors():
    trajectory = {
        "input_ids": [10, 11, 12, 13],
        "loss_mask": [0, 0, 1, 1],
        "logprobs": [0.0, 0.0, -0.1, -0.2],
        "reward": 1.0,
    }

    completion = trajectory_to_completion(trajectory)
    flat, input_lengths = batched_message_log_to_flat_message(
        [completion.message_log], pad_value_dict={"token_ids": 0}
    )

    assert flat["token_ids"].tolist() == [[10, 11, 12, 13]]
    assert flat["token_loss_mask"].tolist() == [[0, 0, 1, 1]]
    assert torch.allclose(
        flat["generation_logprobs"], torch.tensor([[0.0, 0.0, -0.1, -0.2]])
    )
    assert input_lengths.tolist() == [4]
    assert completion.reward == 1.0


@pytest.mark.parametrize("field", ["input_ids", "loss_mask", "logprobs", "reward"])
def test_adapter_rejects_missing_training_field(field):
    trajectory = {
        "input_ids": [10, 11],
        "loss_mask": [0, 1],
        "logprobs": [0.0, -0.1],
        "reward": 1.0,
    }
    trajectory.pop(field)

    with pytest.raises((KeyError, ValueError)):
        trajectory_to_completion(trajectory)
