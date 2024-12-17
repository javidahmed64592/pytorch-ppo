from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class BaseNetwork(nn.Module):
    def __init__(self, models_dir: str) -> None:
        super().__init__()
        self.checkpoint_file = Path(models_dir) / "torch_ppo"

    def save_checkpoint(self) -> None:
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(BaseNetwork):
    def __init__(
        self,
        num_inputs: tuple[int],
        num_outputs: int,
        fc1_dims: int,
        fc2_dims: int,
        alpha: float,
        models_dir: str,
    ) -> None:
        super().__init__(models_dir)
        self.nn = nn.Sequential(
            nn.Linear(*num_inputs, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, num_outputs),
            nn.Softmax(dim=-1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.checkpoint_file = Path(models_dir) / "actor_torch_ppo"

    def forward(self, state: list[float]) -> Categorical:
        dist = self.nn(state)
        return Categorical(dist)


class CriticNetwork(BaseNetwork):
    def __init__(
        self,
        num_inputs: tuple[int],
        fc1_dims: int,
        fc2_dims: int,
        alpha: float,
        models_dir: str,
    ) -> None:
        super().__init__(models_dir)
        self.nn = self.critic = nn.Sequential(
            nn.Linear(*num_inputs, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.checkpoint_file = Path(models_dir) / "critic_torch_ppo"

    def forward(self, state: list[float]) -> Categorical:
        return self.nn(state)
