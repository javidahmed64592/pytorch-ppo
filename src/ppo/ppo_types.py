from pydantic.dataclasses import dataclass


@dataclass
class BaseNetworkType:
    alpha: float
    num_inputs: int
    hidden_layer_sizes: list[int]
    input_shape: tuple[int, int, int] = (0, 0, 0)

    @property
    def models_dir(self) -> str:
        return "models"

    @property
    def name(self) -> str:
        return "base"


@dataclass
class ActorNetworkType(BaseNetworkType):
    num_outputs: int = 0

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
