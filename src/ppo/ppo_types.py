from pydantic.dataclasses import dataclass


@dataclass
class BaseNetworkType:
    models_dir: str
    name: str


@dataclass
class ActorNetworkType(BaseNetworkType):
    num_inputs: tuple[int]
    num_outputs: int
    fc1_dims: int
    fc2_dims: int
    alpha: float


@dataclass
class CriticNetworkType(BaseNetworkType):
    num_inputs: tuple[int]
    fc1_dims: int
    fc2_dims: int
    alpha: float


@dataclass
class AgentType:
    input_dims: tuple[int]
    n_actions: int
    alpha: float
    gamma: float
    gae_lambda: float
    policy_clip: float
    batch_size: int
    n_epochs: int
