import gym
import numpy as np

from ppo.agent import Agent


class Simulation:
    def __init__(self, id: str, n_games: int, max_steps: int) -> None:
        self.env = gym.make(id)
        self.n_games = n_games
        self.max_steps = max_steps
        self.best_score = self.env.reward_range[0]
        self.score_history = []

    def game_loop(self, agent: Agent) -> None:
        avg_score = 0
        n_steps = 0

        for i in range(self.n_games):
            observation, _ = self.env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(observation)
                next_observation, reward, done, _, _ = self.env.step(action)
                n_steps += 1
                score += reward
                agent.remember(observation, action, reward, prob, val, done)
                if n_steps % self.max_steps == 0:
                    agent.learn()
                observation = next_observation
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            if avg_score > self.best_score:
                self.best_score = avg_score
                agent.save_models()

            print(f"Episode {i} | Score: {score:.1f}, Average: {avg_score:.1f} | time_steps: {n_steps}")
