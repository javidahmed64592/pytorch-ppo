from ppo.agent import Agent
from ppo.ppo_types import AgentType
from ppo.simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation("CartPole-v0", n_games=100, max_steps=20)

    agent_config = AgentType(
        input_dims=simulation.env.observation_space.shape,
        n_actions=simulation.env.action_space.n,
        alpha=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=5,
        n_epochs=4,
    )
    agent = Agent(agent_config)
    simulation.run(agent)
