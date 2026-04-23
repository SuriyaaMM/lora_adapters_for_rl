import gymnasium
import torch
import pandas as pd
import torch.nn.functional as F
from typing import Tuple, List
from torch import nn
from torch import distributions
from tqdm import tqdm
from dataclasses import dataclass

@dataclass
class rolloutBuffer:
    observation: torch.Tensor
    reward: torch.Tensor
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor       
    entropy: torch.Tensor
    std: torch.Tensor
    mean: torch.Tensor
    perplexity: torch.Tensor
    done: torch.Tensor

@dataclass
class tunableConfig:
    env_id: str
    epochs_base: int
    epochs_lora: int
    epochs_eval: int
    epochs_ppo: int
    lr_base: float
    lr_lora: float
    n_steps_base: int
    n_steps_lora: int
    perturbation_wind_power: float
    perturbation_wind_turbulence: float
    rank: int
    clip_ratio: float
    critic_loss_weight: float
    entropy_regulariser_weight: float
    gamma: float
    device: torch.device
    seed: int
    train_base: bool
    retrain_base: bool
    train_lora: bool    
    train_base_analytics_file: str 
    train_lora_analytics_file: str
    train_rebase_analytics_file: str
    trained_base_state_dict_file: str
    trained_lora_state_dict_file: str
    trained_rebase_state_dict_file: str
    eval_base_analytics_file: str
    eval_lora_analytics_file: str
    eval_rebase_analytics_file: str
    eval_argmax_base_analytics_file: str
    eval_argmax_lora_analytics_file: str
    eval_argmax_rebase_analytics_file: str
    tb_root: str

