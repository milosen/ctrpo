import gymnasium
import numpy as np
import torch
from gymnasium import Env
from gymnasium import spaces
from typing import Union
from safety_gymnasium.builder import Builder


Array = Union[torch.Tensor, np.ndarray]

class SauteEnv(gymnasium.Wrapper):
    def __init__(
        self, 
        env: Builder,
        safety_budget: float = 25.0, 
        saute_discount_factor: float = 0.99,
        min_rel_budget: float = 1., # minimum relative (with respect to safety_budget) budget
        max_rel_budget: float = 1., # maximum relative (with respect to safety_budget) budget 
        test_rel_budget: float = 1., # test relative budget 
        unsafe_reward: float = 0,
        use_reward_shaping: bool = True,
        use_state_augmentation: bool = True,
        max_ep_len: int = 1000,
    ):
        gymnasium.Wrapper.__init__(self, env)

        assert saute_discount_factor > 0 and saute_discount_factor <= 1, "Please specify a discount factor in (0, 1]" 
        assert max_ep_len > 0

        self.use_reward_shaping = use_reward_shaping
        self.use_state_augmentation = use_state_augmentation
        self.max_ep_len = max_ep_len

        self.obs_high = env.observation_space.high
        self.obs_low = env.observation_space.low
        if self.use_state_augmentation:
            self.obs_high = np.concatenate([self.obs_high, np.array([np.inf], dtype=np.float32)])
            self.obs_low = np.concatenate([self.obs_low, np.array([-np.inf], dtype=np.float32)])
        self.observation_space = spaces.Box(high=self.obs_high, low=self.obs_low)
        self.action_space = env.action_space

        self._saute_discount_factor = saute_discount_factor
        self._unsafe_reward = unsafe_reward
        
        # wrapping the safe environment
        self.wrap = env

        # dealing with safety budget variables
        assert safety_budget > 0, "Please specify a positive safety budget" 
        self.min_rel_budget = min_rel_budget
        self.max_rel_budget = max_rel_budget
        self.test_rel_budget = test_rel_budget
        if self.saute_discount_factor < 1:
            safety_budget = safety_budget  * (1 - self.saute_discount_factor ** self.max_ep_len) / (1 - self.saute_discount_factor) / np.float32(self.max_ep_len)
        self._safety_budget = np.float32(safety_budget) 

        # safety state definition 
        self._safety_state = 1.

    @property
    def safety_budget(self):
        return self._safety_budget 

    @property
    def saute_discount_factor(self):
        return self._saute_discount_factor 

    @property
    def unsafe_reward(self):
        return self._unsafe_reward

    def _augment_state(self, state:np.ndarray, safety_state:np.ndarray):
        """Augmenting the state with the safety state, if needed"""
        augmented_state = np.concatenate([state, np.array([safety_state])], axis=-1) if self.use_state_augmentation else state
        return augmented_state

    def safety_step(self, cost:np.ndarray) -> np.ndarray:
        """ Update the normalized safety state z' = (z - l / d) / gamma. """        
        self._safety_state -= cost / self.safety_budget
        self._safety_state /= self.saute_discount_factor
        return self._safety_state

    def step(self, action):
        """ Step through the environment. """
        next_obs, reward, cost, term, trunc, info = self.wrap.step(action)        
        next_safety_state = self.safety_step(cost)
        info['true_reward'] = reward
        info['next_safety_state'] = np.array([next_safety_state])
        reward = self.reshape_reward(reward, next_safety_state)
        augmented_state  =  self._augment_state(next_obs, next_safety_state)
        return augmented_state, reward, cost, term, trunc, info

    def reset(self, seed=None, options=None) -> np.ndarray:
        """Resets the environment."""
        state, info = self.wrap.reset()
        self._safety_state = self.test_rel_budget
        info['next_safety_state'] = np.array([self._safety_state])
        augmented_state  =  self._augment_state(state,  self._safety_state)    
        return augmented_state, info

    def reshape_reward(self, reward:Array, next_safety_state:Array) -> Array:
        """ Reshaping the reward. """
        if self.use_reward_shaping:
            reward = reward * (next_safety_state > 0) + self.unsafe_reward * (next_safety_state <= 0)
        return reward
    