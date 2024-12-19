from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ppo.ppo_types import ActorNetworkType, BaseNetworkType, CriticNetworkType


class BaseNetwork(nn.Module):
    def __init__(self, config: BaseNetworkType) -> None:
        super().__init__()
        self.checkpoint_file = Path(config.models_dir) / f"{config.name}_torch_ppo"

    def save_checkpoint(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self.checkpoint_file))

    @staticmethod
    def linear_input_layer(num_inputs: int, fc1_dims: int) -> tuple[nn.Linear, nn.ReLU]:
        return nn.Linear(num_inputs, fc1_dims), nn.ReLU()

    @staticmethod
    def linear_hidden_layers(hidden_layer_sizes: list[int]) -> list[nn.Linear]:
        layers = []
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        return layers

    @staticmethod
    def linear_output_layer(num_inputs: int, num_outputs: int) -> nn.Linear:
        return nn.Linear(num_inputs, num_outputs)

    @staticmethod
    def softmax_output_layer(num_inputs: int, num_outputs: int) -> nn.Linear:
        return (
            nn.Linear(num_inputs, num_outputs),
            nn.Softmax(dim=-1),
        )

    @staticmethod
    def convolutional_layers(num_inputs: int) -> list[nn.Conv2d]:
        return [
            nn.Conv2d(num_inputs, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1),
            nn.ReLU(),
        ]


class ActorNetwork(BaseNetwork):
    def __init__(self, config: ActorNetworkType, neural_network: nn.Sequential) -> None:
        super().__init__(config)
        self.nn = neural_network
        self.optimizer = optim.Adam(self.parameters(), lr=config.alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @classmethod
    def from_config(cls, config: ActorNetworkType) -> ActorNetwork:
        neural_network = nn.Sequential(
            *cls.linear_input_layer(config.num_inputs, config.hidden_layer_sizes[0]),
            *cls.linear_hidden_layers(config.hidden_layer_sizes),
            *cls.softmax_output_layer(config.hidden_layer_sizes[-1], config.num_outputs),
        )
        return cls(config, neural_network)

    @classmethod
    def from_config_conv(cls, config: ActorNetworkType) -> ActorNetwork:
        neural_network = nn.Sequential(
            *cls.convolutional_layers(config.num_inputs),
            *cls.linear_hidden_layers(config.hidden_layer_sizes),
            *cls.softmax_output_layer(config.hidden_layer_sizes[-1], config.num_outputs),
        )
        return cls(config, neural_network)

    def forward(self, state: list[float]) -> Categorical:
        dist = self.nn(state)
        return Categorical(dist)

    def calculate_loss(self, weighted_probs: torch.Tensor, weighted_clipped_probs: torch.Tensor) -> torch.Tensor:
        loss = -torch.min(weighted_probs, weighted_clipped_probs)
        return loss.mean()


class CriticNetwork(BaseNetwork):
    def __init__(self, config: CriticNetworkType, neural_network: nn.Sequential) -> None:
        super().__init__(config)
        self.nn = neural_network
        self.optimizer = optim.Adam(self.parameters(), lr=config.alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @classmethod
    def from_config(cls, config: CriticNetworkType) -> CriticNetwork:
        neural_network = nn.Sequential(
            *cls.linear_input_layer(config.num_inputs, config.hidden_layer_sizes[0]),
            *cls.linear_hidden_layers(config.hidden_layer_sizes),
            cls.linear_output_layer(config.hidden_layer_sizes[-1], 1),
        )
        return cls(config, neural_network)

    @classmethod
    def from_config_conv(cls, config: CriticNetworkType) -> CriticNetwork:
        neural_network = nn.Sequential(
            *cls.convolutional_layers(config.num_inputs),
            *cls.linear_hidden_layers(config.hidden_layer_sizes),
            *cls.softmax_output_layer(config.hidden_layer_sizes[-1], 1),
        )
        return cls(config, neural_network)

    def forward(self, state: list[float]) -> Categorical:
        return self.nn(state)

    def calculate_loss(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        value = torch.squeeze(self(states))
        loss = (returns - value) ** 2
        return loss.mean()
