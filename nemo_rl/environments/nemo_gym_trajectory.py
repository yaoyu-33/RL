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
"""Thin adapter from Gym's framework-neutral trajectory to NeMo-RL."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from nemo_rl.experience.interfaces import Completion


SUPPORTED_GYM_TRAJECTORY_SCHEMA_VERSION = 1


def trajectory_to_completion(trajectory: Mapping[str, Any]) -> Completion:
    """Convert one token-aligned Gym trajectory into a NeMo-RL completion.

    Consecutive loss-mask runs become NeMo-RL user/assistant message segments.
    This preserves arbitrary multi-turn masks without making Gym depend on
    NeMo-RL's message-log representation.
    """
    schema_version = trajectory.get("schema_version")
    if schema_version != SUPPORTED_GYM_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Gym trajectory schema_version {schema_version!r}; "
            f"expected {SUPPORTED_GYM_TRAJECTORY_SCHEMA_VERSION}"
        )
    input_ids = trajectory.get("input_ids")
    loss_mask = trajectory.get("loss_mask")
    logprobs = trajectory.get("logprobs")
    reward = trajectory.get("reward")
    if not isinstance(input_ids, Sequence) or not input_ids:
        raise ValueError("NeMo-RL requires a non-empty Gym trajectory input_ids")
    if not isinstance(loss_mask, Sequence) or len(loss_mask) != len(input_ids):
        raise ValueError("Gym trajectory loss_mask must align with input_ids")
    if not isinstance(logprobs, Sequence) or len(logprobs) != len(input_ids):
        raise ValueError("NeMo-RL requires token-aligned Gym trajectory logprobs")
    if reward is None:
        raise ValueError("NeMo-RL requires a reward before adding a Gym trajectory")

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
        env_extras={
            "task_id": trajectory.get("task_id"),
            "sample_id": trajectory.get("sample_id"),
            "metrics": trajectory.get("metrics", {}),
            "metadata": trajectory.get("metadata", {}),
            "messages": trajectory.get("messages", []),
            "steps": trajectory.get("steps", []),
        },
        truncated=bool(trajectory.get("metadata", {}).get("truncated", False)),
        reward=float(reward),
    )
