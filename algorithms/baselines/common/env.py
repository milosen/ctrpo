# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
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
# ==============================================================================
from __future__ import annotations

from typing import Callable
import safety_gymnasium
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeRescaleAction, SafeUnsqueeze
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv

from .wrappers import SafeNormalizeObservation

def make_sa_mujoco_env(num_envs: int, env_id: str, seed: int|None = None):
    """
    Creates and wraps an environment based on the specified parameters.

    Args:
        num_envs (int): Number of parallel environments.
        env_id (str): ID of the environment to create.
        seed (int or None, optional): Seed for the random number generator. Default is None.

    Returns:
        env: The created and wrapped environment.
        obs_space: The observation space of the environment.
        act_space: The action space of the environment.
        
    Examples:
        >>> from safepo.common.env import make_sa_mujoco_env
        >>> 
        >>> env, obs_space, act_space = make_sa_mujoco_env(
        >>>     num_envs=1, 
        >>>     env_id="SafetyPointGoal1-v0", 
        >>>     seed=0
        >>> )
    """
    if num_envs > 1:
        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            env = safety_gymnasium.make(env_id)
            env = SafeRescaleAction(env, -1.0, 1.0)
            return env
        env_fns = [create_env for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns)
        env = SafeNormalizeObservation(env)
        env.reset(seed=seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = safety_gymnasium.make(env_id)
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)
    
    return env, obs_space, act_space
