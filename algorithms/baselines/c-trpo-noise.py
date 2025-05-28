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

import os
import random
import sys
import time
from collections import deque
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from common.buffer import VectorizedOnPolicyBuffer, discount_cumsum
from common.env import make_sa_mujoco_env, make_sa_isaac_env
from common.logger import EpochLogger
from common.model import ActorVCritic
from common.utils.config import single_agent_args, isaac_gym_map, parse_sim_params

CONJUGATE_GRADIENT_ITERS=15
TRPO_SEARCHING_STEPS=15

default_cfg = {
    'hidden_sizes': [64, 64],
    'gamma': 0.99,
    'target_kl': 0.01,
    'batch_size': 256,
    'learning_iters': 10,
    'max_grad_norm': 40.0,
    'beta': 1.0,  # 0.05
    'adaptive_target_kl': False,
    'positive_cost': False,
    'hyst_frac': 1.0,
    'noise_level': 0.5,
}

env_configs = {
    "SafetyAntVelocity-v1": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyHalfCheetahVelocity-v1": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyHumanoidVelocity-v1": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyHopperVelocity-v1": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyCarButton1-v0": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyPointGoal1-v0": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyRacecarCircle1-v0": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
    "SafetyAntPush1-v0": {
        'hidden_sizes': default_cfg["hidden_sizes"],
        'gamma': 0.99,
        'target_kl': default_cfg["target_kl"],
        'batch_size': 128,
        'learning_iters': 10,
        'max_grad_norm': 40.0,
        'beta': default_cfg["beta"],
        'adaptive_target_kl': default_cfg["adaptive_target_kl"],
        'positive_cost': default_cfg["positive_cost"],
        'hyst_frac': default_cfg["hyst_frac"],
        'noise_level': default_cfg["noise_level"],
    },
}

isaac_gym_specific_cfg = {
    'total_steps': 100000000,
    'steps_per_epoch': 32768,
    'hidden_sizes': [1024, 1024, 512],
    'gamma': 0.96,
    'target_kl': default_cfg["target_kl"],
    'num_mini_batch': 4,
    'use_value_coefficient': True,
    'learning_iters': 8,
    'max_grad_norm': 1.0,
    'use_critic_norm': False,
    'beta': default_cfg["beta"],
    'adaptive_target_kl': default_cfg["adaptive_target_kl"],
    'positive_cost': default_cfg["positive_cost"],
    'hyst_frac': default_cfg["hyst_frac"],
}


def build_custom_mlp_network(sizes):
    """
    Build a multi-layer perceptron (MLP) neural network.

    This function constructs an MLP network with the specified layer sizes and activation functions.

    Args:
        sizes (list of int): List of integers representing the sizes of each layer in the network.

    Returns:
        nn.Sequential: An instance of PyTorch's Sequential module representing the constructed MLP.
    """
    layers = list()
    for j in range(len(sizes) - 1):
        act = nn.ReLU if j < len(sizes) - 2 else nn.ReLU
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, "No gradients were found in model parameters."
    return torch.cat(flat_params)


def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    vector_b: torch.Tensor,
    data, args, ep_costs, config, optim_case,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs, data, args, ep_costs, config, optim_case)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p, policy, fvp_obs, data, args, ep_costs, config, optim_case)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"


def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, "No gradients were found in model parameters."
    return torch.cat(grads)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def lambda_return(data, gamma):
    # todo: implement lambda-returns (now it's just MC)
    costs = data["cost"]
    dones = data["done"]
    indx = torch.cat([torch.tensor([[0]], device=dones.device), dones.nonzero() + 1])
    c_ret_list = []
    for i1, i2 in zip(indx[:-1], indx[1:]):
        ep_costs = costs[i1:i2].clone()
        c_ret_list.append(data['target_value_c'][0].item())
    return sum(c_ret_list)/len(c_ret_list)


DIV_MODE = "entropy"


def cost_barrier_divergence(pi_old, pi_new, data, beta: float, cost_limit: float, ep_costs: float, gamma: float, optim_case: int):
    barrier_div = torch.distributions.kl.kl_divergence(pi_old, pi_new).mean()

    if optim_case == 0:  # recovery
        return barrier_div
    elif optim_case == 1:
        c_advantages = data['adv_c']

        log_prob = pi_new.log_prob(data["act"]).sum(dim=-1)
        
        # the pi_new/pi_old-1 is to make sure D_c = 0 when pi_new = pi_old
        alpha = 1/(1-gamma)*((torch.exp(log_prob - data["log_prob"]) - 1)*data['adv_c']).mean()

        b = cost_limit
        delta_b = b - ep_costs
        assert 0 < delta_b <= b  # equal to b when cost return is zero
        
        if not alpha < delta_b:
            return torch.tensor(torch.inf)

        if DIV_MODE == "barrier":
            phi = -(torch.log(1/delta_b*(delta_b - alpha)) + 1/delta_b*alpha)
            
        elif DIV_MODE == "entropy":    
            phi = alpha + (delta_b - alpha)*torch.log(1/delta_b*(delta_b - alpha))

        barrier_term = beta*phi

        barrier_div += barrier_term

        return barrier_div
    else:
        raise ValueError(f"Optim case {optim_case} unknown")


def fvp(
    params: torch.Tensor,
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    data, args, ep_costs, config, optim_case,
) -> torch.Tensor:
    policy.actor.zero_grad()
    current_distribution = policy.actor(fvp_obs)
    with torch.no_grad():
        old_distribution = policy.actor(fvp_obs)

    div = cost_barrier_divergence(
        old_distribution, current_distribution, data, 
        beta=config['beta'], cost_limit=args.cost_limit, ep_costs=ep_costs, gamma=config['gamma'], optim_case=optim_case
    )

    grads = torch.autograd.grad(div, tuple(policy.actor.parameters()), create_graph=True)
    flat_grad_div = torch.cat([grad.view(-1) for grad in grads])

    div_p = (flat_grad_div * params).sum()
    grads = torch.autograd.grad(
        div_p,
        tuple(policy.actor.parameters()),
        retain_graph=False,
    )

    flat_grad_grad_div = torch.cat([grad.contiguous().view(-1) for grad in grads])

    return flat_grad_grad_div + params * 0.1


class NoisyBuffer(VectorizedOnPolicyBuffer):
    def __init__(self, *args, **kwargs):
        self.noise_level = kwargs.pop("noise_level", 0.1)
        super().__init__(*args, **kwargs)

    def store(self, obs, act, reward, cost, value_r, value_c, log_prob):
        noise = torch.randn_like(value_c) * self.noise_level
        value_c = value_c + noise
        super().store(obs=obs, act=act, reward=reward, cost=cost, value_r=value_r, value_c=value_c, log_prob=log_prob)
    
    def finish_path(self, last_value_r = None, last_value_c = None, idx = 0):
        noise = torch.randn_like(last_value_c) * self.noise_level
        last_value_c = last_value_c + noise
        return super().finish_path(last_value_r, last_value_c, idx)


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')


    if args.task not in isaac_gym_map.keys():
        env, obs_space, act_space = make_sa_mujoco_env(
            num_envs=args.num_envs, env_id=args.task, seed=args.seed
        )
        eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args.task, seed=None)
        config = env_configs.get(args.task, default_cfg)

    else:
        sim_params = parse_sim_params(args, cfg_env, None)
        env = make_sa_isaac_env(args=args, cfg=cfg_env, sim_params=sim_params)
        eval_env = env
        obs_space = env.observation_space
        act_space = env.action_space
        args.num_envs = env.num_envs
        config = isaac_gym_specific_cfg

    # set training steps
    steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
    total_steps = config.get("total_steps", args.total_steps)
    local_steps_per_epoch = steps_per_epoch // args.num_envs
    epochs = total_steps // steps_per_epoch
    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
        hidden_sizes=config["hidden_sizes"],
    ).to(device)

    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=1e-3
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=1e-3
    )
    # create the vectorized on-policy buffer
    buffer = NoisyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
        gamma=config["gamma"],
        noise_level=config["noise_level"],
    )
    print("noise level: ", config["noise_level"])

    # set up the logger
    dict_args = vars(args)
    dict_args.update(config)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
    )
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")
    obs, _ = env.reset()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ep_ret, ep_cost, ep_len = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
    )
    optim_case = 0
    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
            next_obs, reward, cost, terminated, truncated, info = env.step(action)

            ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
            ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            epoch_end = steps >= local_steps_per_epoch - 1
            
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCost": np.mean(cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                                "Metrics/EpCostRegret": np.clip(np.array(cost_deque) - args.cost_limit, 
                                                                a_min=0, a_max=None).sum()
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0
                        logger.logged = False

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost": np.mean(eval_cost),
                    "Metrics/EvalEpLen": np.mean(eval_len),  # np.count_nonzero(eval_cost < args.cost_limit)
                }
            )

        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCost")

        # update policy
        data = buffer.get()
        fvp_obs = data["obs"][:: 1]
        theta_old = get_flat_params_from(policy.actor)
        policy.actor.zero_grad()
        
        # compute loss_pi
        temp_distribution = policy.actor(data["obs"])
        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        
        if ((optim_case == 1) and (ep_costs < args.cost_limit)) or ((optim_case == 0) and (ep_costs < config["hyst_frac"]*args.cost_limit)):
            # normal loss
            optim_case = 1
            loss_pi = -(ratio * data["adv_r"]).mean()
        else:
            # recovery loss
            optim_case = 0
            loss_pi = (ratio * data["adv_c"]).mean()
        
        loss_before = loss_pi.item()
        old_distribution = policy.actor(data["obs"])

        loss_pi.backward()

        grads = -get_flat_gradients_from(policy.actor)
        
        x = conjugate_gradients(fvp, policy, fvp_obs, grads, data, args, ep_costs, config, optim_case, CONJUGATE_GRADIENT_ITERS)
        
        assert torch.isfinite(x).all(), "x is not finite"
        xHx = torch.dot(x, fvp(x, policy, fvp_obs, data, args, ep_costs, config, optim_case))
        assert xHx.item() >= 0, "xHx is negative"
        alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), "step_direction is not finite"

        step_frac = 1.0
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = grads.dot(step_direction)

        final_div = 0.0

        # While not within_trust_region and not out of total_steps:
        for step in range(TRPO_SEARCHING_STEPS):
            # update theta params
            new_theta = theta_old + step_frac * step_direction
            # set new params as params of net
            set_param_values_to_model(policy.actor, new_theta)

            with torch.no_grad():
                temp_distribution = policy.actor(data["obs"])
                log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["log_prob"])
    
                # compute KL distance between new and old policy
                current_distribution = policy.actor(data["obs"])
                
                div = cost_barrier_divergence(old_distribution, current_distribution, data, 
                                              beta=config['beta'], cost_limit=args.cost_limit, ep_costs=ep_costs,
                                              gamma=config["gamma"], optim_case=optim_case).item()
                
                kl = torch.distributions.kl.kl_divergence(old_distribution, current_distribution).mean().item()
                
                if optim_case > 0:
                    loss_pi = -(ratio * data["adv_r"]).mean()
                    target_div = (1-0.5*ep_costs/args.cost_limit)*config["target_kl"] if config["adaptive_target_kl"] else config["target_kl"]
                else:
                    loss_pi = (ratio * data["adv_c"]).mean()
                    target_div = config["target_kl"]
                
            # real loss improve: old policy loss - new policy loss
            loss_improve = loss_before - loss_pi.item()
            logger.log(
                f"Expected Improvement: {expected_improve} Actual: {loss_improve}"
            )
            if not torch.isfinite(loss_pi):
                logger.log("WARNING: loss_pi not finite")
            elif (loss_improve < 0):
                logger.log("INFO: did not improve <0")
            elif div > target_div:
                logger.log("INFO: violated KL constraint.")
            else:
                # step only if surrogate is improved and when within trust reg.
                acceptance_step = step + 1
                logger.log(f"Accept step at i={acceptance_step}")
                final_div = div
                break
            step_frac *= 0.8
        else:
            logger.log("INFO: no suitable step found...")
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        theta_new = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_new)

        logger.store(
            **{
                "Misc/Alpha": alpha.item(),
                "Misc/FinalStepNorm": torch.norm(step_direction).mean().item(),
                "Misc/xHx": xHx.item(),
                "Misc/gradient_norm": torch.norm(grads).mean().item(),
                "Misc/H_inv_g": x.norm().item(),
                "Misc/AcceptanceStep": acceptance_step,
                "Misc/optim_case": optim_case,
                "Loss/Loss_actor": loss_pi.mean().item(),
                "Train/KL": kl,
                "Train/div": final_div,
                "Train/target_div": target_div,
            },
        )

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["cost"],
                data["target_value_r"],
                data["target_value_c"],
            ),
            batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
            shuffle=True,
        )
        for _ in range(config["learning_iters"]):
            for (
                obs_b, act_b, cost_b,
                target_value_r_b,
                target_value_c_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                if config.get("use_critic_norm", True):
                    for param in policy.reward_critic.parameters():
                        loss_r += param.pow(2).sum() * 0.001
                    for param in policy.cost_critic.parameters():
                        loss_c += param.pow(2).sum() * 0.001
                total_loss = 2*loss_r + loss_c \
                    if config.get("use_value_coefficient", False) \
                    else loss_r + loss_c
                total_loss.backward()
                clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                reward_critic_optimizer.step()
                cost_critic_optimizer.step()

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                    }
                )
        update_end_time = time.time()
        if not logger.logged:
            # log data
            logger.log_tabular("Metrics/EpRet")
            logger.log_tabular("Metrics/EpCost")
            logger.log_tabular("Metrics/EpLen")
            logger.log_tabular("Metrics/EpCostRegret")
            if args.use_eval:
                logger.log_tabular("Metrics/EvalEpRet")
                logger.log_tabular("Metrics/EvalEpCost")
                logger.log_tabular("Metrics/EvalEpLen")
            logger.log_tabular("Train/Epoch", epoch + 1)
            logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
            logger.log_tabular("Train/KL")
            logger.log_tabular("Train/div")
            logger.log_tabular("Train/target_div")
            logger.log_tabular("Loss/Loss_reward_critic")
            logger.log_tabular("Loss/Loss_cost_critic")
            logger.log_tabular("Loss/Loss_actor")
            logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
            if args.use_eval:
                logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
            logger.log_tabular("Time/Update", update_end_time - eval_end_time)
            logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
            logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
            logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())
            logger.log_tabular("Misc/Alpha")
            logger.log_tabular("Misc/FinalStepNorm")
            logger.log_tabular("Misc/xHx")
            logger.log_tabular("Misc/gradient_norm")
            logger.log_tabular("Misc/H_inv_g")
            logger.log_tabular("Misc/AcceptanceStep")
            logger.log_tabular("Misc/optim_case")

            logger.dump_tabular()
            if (epoch+1) % 100 == 0 or epoch == 0:
                logger.torch_save(itr=epoch)
                if args.task not in isaac_gym_map.keys():
                    logger.save_state(
                        state_dict={
                            "Normalizer": env.obs_rms,
                        },
                        itr = epoch
                    )
    logger.close()


if __name__ == "__main__":
    args, cfg_env = single_agent_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args, cfg_env)
    else:
        main(args, cfg_env)
