import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class policyNetwork(nn.Module):
    def __make_network(
        self, 
        in_features: int,
        out_features: int
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(
                in_features=in_features, 
                out_features=1024, 
                device=self.device
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=1024, 
                out_features=2048, 
                device=self.device
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=2048, 
                out_features=512, 
                device=self.device
            ),
            nn.SiLU(),
            nn.Linear(
                in_features=512, 
                out_features=out_features, 
                device=self.device
            )
        )

    def __init__(
        self, 
        observation_shape: int, 
        action_shape: int, 
        device: torch.device
    ):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.device = device

        self.actor_net = self.__make_network(
            in_features=self.observation_shape, 
            out_features=self.action_shape
        )
        self.critic_net = self.__make_network(
            in_features=self.observation_shape, 
            out_features=1
        )
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        logits = self.actor_net(x)
        value = self.critic_net(x)

        return logits, value

class policyNetworkLoRA(nn.Module):
    def __init__(
        self, 
        policy_network: policyNetwork, 
        rank: int
    ):
        super().__init__()
        self.policy_network = policy_network
        self.policy_network.requires_grad_(False)
        self.rank = rank

        self.a = nn.Parameter(
            data=torch.randn(
                size=(policy_network.observation_shape, rank), 
                dtype=torch.float32, 
                device=policy_network.device
            ) * 0.01,
            requires_grad=True
        )
        self.ca = nn.Parameter(
            data=torch.randn(
                size=(policy_network.observation_shape, rank), 
                dtype=torch.float32, 
                device=policy_network.device
            ) * 0.01,
            requires_grad=True
        )
        self.b = nn.Parameter(
            data=torch.zeros(
                size=(rank, policy_network.action_shape), 
                dtype=torch.float32, 
                device=policy_network.device
            ),
            requires_grad=True
        )
        self.cb = nn.Parameter(
            data=torch.zeros(
                size=(rank, 1), 
                dtype=torch.float32, 
                device=policy_network.device
            ),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # LoRA adapters (trainable, builds computation graph)
        xa = x @ self.a
        xc = x @ self.ca
        lora_logits = (F.gelu(F.silu(xa))) @ self.b
        lora_value = (F.gelu(F.silu(xc))) @ self.cb
        
        # Base network (completely frozen, NO graph, NO activation caching)
        with torch.no_grad():
            base_logits, base_value = self.policy_network(x)
        
        return lora_logits + base_logits, lora_value + base_value
