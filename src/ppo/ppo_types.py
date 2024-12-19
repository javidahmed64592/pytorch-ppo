from pydantic.dataclasses import dataclass


@dataclass
class BaseNetworkType:
    alpha: float
    num_inputs: int
    fc1_dims: int
    fc2_dims: int

    @property
    def models_dir(self) -> str:
        return "models"

    @property
    def name(self) -> str:
        return "base"


@dataclass
class ActorNetworkType(BaseNetworkType):
    num_outputs: int

    @property
    def name(self) -> str:
        return "agent"


@dataclass
class CriticNetworkType(BaseNetworkType):
    @property
    def name(self) -> str:
        return "critic"


@dataclass
class AgentType:
    gamma: float
    gae_lambda: float
    policy_clip: float
    batch_size: int
    n_epochs: int
