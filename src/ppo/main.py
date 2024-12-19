import numpy as np

from ppo.agent import Agent
from ppo.ppo_types import ActorNetworkType, AgentType, CriticNetworkType
from ppo.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation("CartPole-v1", n_games=100, max_steps=20)
    try:
        num_inputs = int(np.prod(simulation.env.observation_space.shape))
    except TypeError:
        num_inputs = len(simulation.env.observation_space)
    num_actions = simulation.env.action_space.n

    agent_config = AgentType(
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=5,
        n_epochs=4,
    )
    actor_config = ActorNetworkType(
        alpha=3e-4,
        num_inputs=num_inputs,
        hidden_layer_sizes=[256, 256],
        num_outputs=num_actions,
    )
    critic_config = CriticNetworkType(
        alpha=3e-4,
        num_inputs=num_inputs,
        hidden_layer_sizes=[256, 256],
    )
    agent = Agent(agent_config, actor_config, critic_config)
    simulation.run(agent)
