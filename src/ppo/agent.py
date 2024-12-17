import numpy as np
import torch
from numpy.typing import NDArray

from ppo.networks import ActorNetwork, CriticNetwork

rng = np.random.default_rng()


class PPOMemory:
    def __init__(self, batch_size: int) -> None:
        self.clear_memory()
        self.batch_size = batch_size

    def generate_batches(self) -> tuple[NDArray]:
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int8)
        rng.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.dones),
            batches,
        )

    def store_memory(
        self, state: list[float], action: list[float], reward: float, probs: list[float], vals: list[float], done: bool
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.probs.append(probs)
        self.vals.append(vals)
        self.dones.append(done)

    def clear_memory(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.probs = []
        self.vals = []
        self.dones = []


class Agent:
    def __init__(
        self,
        input_dims: tuple[int],
        n_actions: int,
        alpha: float,
        gamma: float,
        gae_lambda: float,
        policy_clip: float,
        batch_size: int,
        n_epochs: int,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.actor = ActorNetwork(input_dims, n_actions, 256, 256, alpha, "models")
        self.critic = CriticNetwork(input_dims, 256, 256, alpha, "models")
        self.memory = PPOMemory(batch_size)

    def remember(
        self, state: list[float], action: list[float], reward: float, probs: list[float], vals: list[float], done: bool
    ) -> None:
        self.memory.store_memory(state, action, reward, probs, vals, done)

    def save_models(self) -> None:
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self) -> None:
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation: list[float]) -> list[float]:
        state = torch.tensor([observation], dtype=torch.float).to(self.actor.device)
        dist = self.actor(state)
        action = dist.sample()
        probs = dist.log_prob(action)
        value = self.critic(state)

        action = torch.squeeze(action).item()
        probs = torch.squeeze(probs).item()
        value = torch.squeeze(value).item()
        return action, probs, value

    def backpropagate_loss(self, actor_loss: torch.Tensor, critic_loss: torch.Tensor) -> None:
        total_loss = actor_loss + (0.5 * critic_loss)
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def learn(self) -> None:
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                reward_arr,
                old_probs_arr,
                vals_arr,
                done_arr,
                batches,
            ) = self.memory.generate_batches()
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (
                        reward_arr[k] + self.gamma * vals_arr[k + 1] * (1 - int(done_arr[k])) - vals_arr[k]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t

            advantage = torch.tensor(advantage).to(self.actor.device)
            values = torch.tensor(vals_arr).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)
                old_probs = torch.tensor(old_probs_arr[batch]).to(self.actor.device)
                returns = advantage[batch] + values[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)
                prob_ratio = self.calculate_prob_ratio(old_probs, new_probs)

                weighted_probs, weighted_clipped_probs = self.calculate_weighted_probs(advantage[batch], prob_ratio)

                actor_loss = self.actor.calculate_loss(weighted_probs, weighted_clipped_probs)
                critic_loss = self.critic.calculate_loss(states, returns)

                self.backpropagate_loss(actor_loss, critic_loss)

        self.memory.clear_memory()

    def calculate_prob_ratio(self, old_probs: torch.Tensor, new_probs: torch.Tensor) -> torch.Tensor:
        return new_probs.exp() / old_probs.exp()

    def calculate_weighted_probs(self, advantage: torch.Tensor, prob_ratio: torch.Tensor) -> torch.Tensor:
        weighted_probs = advantage * prob_ratio
        weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage
        return weighted_probs, weighted_clipped_probs
