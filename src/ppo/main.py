from ppo.agent import Agent
from ppo.ppo_types import ActorNetworkType, AgentType, CriticNetworkType
from ppo.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation("CartPole-v1", n_games=100, max_steps=20)

    agent_config = AgentType(
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=5,
        n_epochs=4,
    )
    actor_config = ActorNetworkType(
        alpha=3e-4,
        num_inputs=simulation.env.observation_space.shape,
        fc1_dims=256,
        fc2_dims=256,
        num_outputs=simulation.env.action_space.n,
    )
    critic_config = CriticNetworkType(
        alpha=3e-4,
        num_inputs=simulation.env.observation_space.shape,
        fc1_dims=256,
        fc2_dims=256,
    )
    agent = Agent(agent_config, actor_config, critic_config)
    simulation.run(agent)
