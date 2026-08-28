# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Gym's training trajectory to NeMo-RL tensors."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from nemo_rl.experience.interfaces import Completion


def trajectory_to_completion(trajectory: Mapping[str, Any]) -> Completion:
    """Split a token-aligned Gym trajectory into trainable message runs."""
    input_ids = trajectory["input_ids"]
    loss_mask = trajectory["loss_mask"]
    logprobs = trajectory["logprobs"]
    if not isinstance(input_ids, Sequence) or not input_ids:
        raise ValueError("NeMo-RL requires a non-empty Gym trajectory input_ids")
    if not isinstance(loss_mask, Sequence) or len(loss_mask) != len(input_ids):
        raise ValueError("Gym trajectory loss_mask must align with input_ids")
    if not isinstance(logprobs, Sequence) or len(logprobs) != len(input_ids):
        raise ValueError("NeMo-RL requires token-aligned Gym trajectory logprobs")
    message_log: list[dict[str, Any]] = []
    start = 0
    for end in range(1, len(input_ids) + 1):
        if end < len(input_ids) and loss_mask[end] == loss_mask[start]:
            continue
        trainable = loss_mask[start] == 1
        if loss_mask[start] not in (0, 1):
            raise ValueError("Gym trajectory loss_mask values must be 0 or 1")
        message: dict[str, Any] = {
            "role": "assistant" if trainable else "user",
            "content": "",
            "token_ids": torch.tensor(input_ids[start:end], dtype=torch.long),
            "token_loss_mask": torch.full(
                (end - start,), int(trainable), dtype=torch.long
            ),
            "generation_logprobs": torch.tensor(
                logprobs[start:end] if trainable else [0.0] * (end - start),
                dtype=torch.float32,
            ),
        }
        message_log.append(message)
        start = end

    return Completion(
        message_log=message_log,  # type: ignore[arg-type]
        env_extras={},
        truncated=False,
        reward=float(trajectory["reward"]),
    )
