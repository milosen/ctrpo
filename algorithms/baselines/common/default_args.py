from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class CommonArgs:
    task: str = "SafetyPointGoal1-v0"
    """The task to run"""
    seed: int = 0  
    """Random seed"""
    use_eval: bool = False
    """Use evaluation environment for testing"""
    print_tasks: bool = False
    """Print available tasks and exit"""
    num_envs: int = 10
    """The number of parallel game environments"""
    experiment: str = "single_agent_exp"
    """Experiment name"""
    log_dir: str = "runs"
    """Directory to save agent logs"""
    device: str = "cpu"
    """The device to run the model on"""
    device_id: int = 0
    """The device id to run the model on"""
    write_terminal: bool = True
    """Toggles terminal logging"""
    headless: bool = False
    """Toggles headless mode"""
    total_steps: int = 10000000
    """Total timesteps of the experiments"""
    steps_per_epoch: int = 5000
    """The number of steps to run in each environment per policy rollout"""
    randomize: bool = False
    """Whether to randomize the environments' initial states"""
    cost_limit: float = 25.0
    """Cost Limit"""
    hidden_sizes: list = field(default_factory=lambda: [64, 64])
    """Base network architecture for each component."""
    gamma: float = 0.99
    """The discount factor."""
    gae_lambda: float = 0.95
    """The discount factor."""
    batch_size: int = 64
    """The batch size for mini-batch training."""
    learning_iters: int = 40
    """Training iterations for value networks, i.e. number of dataset passes."""
    max_grad_norm: float = 40.0
    """The max grad norm."""
    use_critic_norm: bool = True
    """Use critic normalization."""
    value_coefficient: float = 1.
    """Multiply reward value loss."""
    learning_rate: float = 3e-4
    """The learning rate of all NN components."""
    debug: bool = False
    """Print debugging info."""

@dataclass
class AlgoArgs:
    pass

@dataclass
class Args:
    algos: AlgoArgs
    common: CommonArgs