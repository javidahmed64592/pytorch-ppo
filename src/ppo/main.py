import gym
import matplotlib.pyplot as plt
import numpy as np

from ppo.agent import Agent


def plot_learning_curve(x: list[float], scores: list[float], figure_file: str) -> None:
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100) : (i + 1)])
    plt.plot(x, running_avg)
    plt.title("Running average of previous 100 scores")
    plt.savefig(figure_file)


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    max_steps = 20

    alpha = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    policy_clip = 0.2
    batch_size = 5
    n_epochs = 4

    n_games = 100
    figure_file = "cartpole.png"
    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    agent = Agent(
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            next_observation, reward, done, _, _ = env.step(action)
            n_steps += 1
            score += reward
            agent.remember(observation, action, reward, prob, val, done)
            if n_steps % max_steps == 0:
                agent.learn()
                learn_iters += 1
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print(
            f"Episode {i} | Score: {score:.1f}, Average: {avg_score:.1f} | time_steps: {n_steps} | learning_steps: {learn_iters}"
        )

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
