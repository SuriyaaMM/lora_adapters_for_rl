import gymnasium
import torch
import nvtx
import pandas as pd
import torch.nn.functional as F
import torch.profiler as profiler
from typing import Tuple, List
from torch import nn
from torch import distributions
from tqdm import tqdm
from dataclasses import dataclass

from source.utils import *
from source.models import *
from source.helpers import *

def main(config: tunableConfig):
    device = config.device

    env = gymnasium.make(id=config.env_id)
    perturbed_env = gymnasium.make(
        id=config.env_id,
        enable_wind=True,
        wind_power=config.perturbation_wind_power,
        turbulence_power=config.perturbation_wind_turbulence
    )

    policy_network_base = policyNetwork(
        observation_shape=env.observation_space.shape[0],
        action_shape=env.action_space.n,
        device=device
    )
    optimiser_base = torch.optim.AdamW(
        params=policy_network_base.parameters(),
        lr=config.lr_base
    )

    with nvtx.annotate("BASE_TRAIN", color="red", domain="BASE"):
        train(
            config=config,
            epochs=config.epochs_base,
            n_steps=config.n_steps_base,
            env=env,
            policy_model=policy_network_base,
            optimiser=optimiser_base,
            desc="training base model",
            colour="magenta",
            train_analytics_savefile=config.train_base_analytics_file,
            eval_analytics_savefile=config.eval_base_analytics_file,
            eval_argmax_analytics_savefile=config.eval_argmax_base_analytics_file,
            policy_model_savefile=config.trained_base_state_dict_file,
            eval_only=False,
        )
        torch.cuda.synchronize()

    with nvtx.annotate("MEMORY_FENCE", color="yellow", domain="BASE"):
        del optimiser_base
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    policy_network_lora = policyNetworkLoRA(
        policy_network=policy_network_base,
        rank=config.rank
    )
    optimiser_lora = torch.optim.AdamW(
        params=policy_network_lora.parameters(),
        lr=config.lr_lora
    )

    with nvtx.annotate("LORA_TRAIN", color="blue", domain="LORA"):
        train(
            config=config,
            epochs=config.epochs_lora,
            n_steps=config.n_steps_lora,
            env=perturbed_env,
            policy_model=policy_network_lora,
            optimiser=optimiser_lora,
            desc="training LoRA model",
            colour="green",
            train_analytics_savefile=config.train_lora_analytics_file,
            eval_analytics_savefile=config.eval_lora_analytics_file,
            eval_argmax_analytics_savefile=config.eval_argmax_lora_analytics_file,
            policy_model_savefile=config.trained_lora_state_dict_file,
            eval_only=False,
        )
        torch.cuda.synchronize()

if __name__ == "__main__":
    tunable_config = tunableConfig(
        env_id="LunarLander-v3",
        epochs_base=1,
        epochs_eval=1,
        epochs_lora=1,
        epochs_ppo=1,
        lr_base=3e-4,
        lr_lora=3e-3,
        n_steps_base=8000,
        n_steps_lora=8000,
        perturbation_wind_power=15.0,
        perturbation_wind_turbulence=1.5,
        rank=12,
        clip_ratio=0.30,
        critic_loss_weight=0.5,
        entropy_regulariser_weight=0.01,
        gamma=0.99,
        device=torch.device("cuda"),
        seed=42,
        train_base=True,
        retrain_base=True,
        train_lora=True,
        train_base_analytics_file="data/base_analytics3.csv",
        train_lora_analytics_file="data/lora_analytics3.csv",
        train_rebase_analytics_file="data/rebase_analytics3.csv",
        trained_base_state_dict_file="models/policy_network_base3.pth",
        trained_lora_state_dict_file="models/policy_network_lora3.pth",
        trained_rebase_state_dict_file="models/policy_network_rebase3.pth",
        eval_base_analytics_file="data/eval_base_analytics3.csv",
        eval_lora_analytics_file="data/eval_lora_analytics3.csv",
        eval_rebase_analytics_file="data/eval_rebase_analytics3.csv",
        eval_argmax_base_analytics_file="data/eval_argmax_base_analytics3.csv",
        eval_argmax_lora_analytics_file="data/eval_argmax_lora_analytics3.csv",
        eval_argmax_rebase_analytics_file="data/eval_argmax_rebase_analytics3.csv",
        tb_root="traces/test_1"
    )

    main(config=tunable_config)