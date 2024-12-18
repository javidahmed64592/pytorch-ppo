import gymnasium as gym
import numpy as np
import torch

from ppo.agent import Agent


class Simulation:
    def __init__(self, id: str, n_games: int, max_steps: int) -> None:
        self.env = gym.make(id, render_mode="human")
        self.n_games = n_games
        self.max_steps = max_steps
        self.timesteps = 0

    def step(self, action: torch.Tensor) -> tuple[float, float, bool]:
        next_observation, reward, done, _, _ = self.env.step(action)
        return next_observation, reward, done

    def game_loop(self, agent: Agent) -> None:
        observation, _ = self.env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            next_observation, reward, done = self.step(action)
            self.timesteps += 1
            score += reward
            agent.memory.store_memory(observation, action, reward, prob, val, done)
            if self.timesteps % self.max_steps == 0:
                agent.learn()
            observation = next_observation
        return score

    def run(self, agent: Agent) -> None:
        score_history = []
        avg_score = 0
        best_score = 0

        for i in range(self.n_games):
            score = self.game_loop(agent)
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print(f"Episode {i} | Score: {score:.1f}, Average: {avg_score:.1f}")

        self.env.close()
