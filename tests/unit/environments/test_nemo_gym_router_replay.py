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

import asyncio

import pytest

from nemo_rl.environments.nemo_gym import NemoGym


def test_minimal_trajectory_demo_rejects_router_replay():
    class _MockSelf:
        cfg = {"require_routed_experts": True}
        rh = object()

        def _require_spinup(self):
            pass

    stream = NemoGym.__ray_metadata__.modified_class.run_rollouts(
        _MockSelf(), [{"_rowidx": 0}], "test"
    )
    with pytest.raises(NotImplementedError, match="text rollouts only"):
        asyncio.run(stream.__anext__())
