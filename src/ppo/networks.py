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

    def linear_input_layer(self, num_inputs: int, fc1_dims: int) -> tuple[nn.Linear, nn.ReLU]:
        return nn.Linear(num_inputs, fc1_dims), nn.ReLU()

    def linear_hidden_layers(self, hidden_layer_sizes: list[int]) -> list[nn.Linear]:
        layers = []
        for i in range(len(hidden_layer_sizes) - 1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        return layers


class ActorNetwork(BaseNetwork):
    def __init__(self, config: ActorNetworkType) -> None:
        super().__init__(config)
        self.nn = nn.Sequential(
            *self.linear_input_layer(config.num_inputs, config.hidden_layer_sizes[0]),
            *self.linear_hidden_layers(config.hidden_layer_sizes),
            nn.Linear(config.hidden_layer_sizes[-1], config.num_outputs),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=config.alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: list[float]) -> Categorical:
        dist = self.nn(state)
        return Categorical(dist)

    def calculate_loss(self, weighted_probs: torch.Tensor, weighted_clipped_probs: torch.Tensor) -> torch.Tensor:
        loss = -torch.min(weighted_probs, weighted_clipped_probs)
        return loss.mean()


class CriticNetwork(BaseNetwork):
    def __init__(
        self,
        config: CriticNetworkType,
    ) -> None:
        super().__init__(config)
        self.nn = nn.Sequential(
            *self.linear_input_layer(config.num_inputs, config.hidden_layer_sizes[0]),
            *self.linear_hidden_layers(config.hidden_layer_sizes),
            nn.Linear(config.hidden_layer_sizes[-1], 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=config.alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state: list[float]) -> Categorical:
        return self.nn(state)

    def calculate_loss(self, states: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        value = torch.squeeze(self(states))
        loss = (returns - value) ** 2
        return loss.mean()
