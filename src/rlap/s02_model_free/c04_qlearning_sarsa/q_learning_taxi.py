"""
The idea of Q-learning is to approximate the Q-function by using the current optimal action value.
The Q-learning update is very similar to the update done in SARSA,
with the exception that it takes the maximum state-action value
"""

import numpy as np
import gym


def eps_greedy(Q, s, eps=0.1):
    """
    Epsilon greedy policy
    """
    if np.random.uniform(0, 1) < eps:
        # Choose a random action
        return np.random.randint(Q.shape[1])
    else:
        # Choose the action of a greedy policy
        return greedy(Q, s)


def greedy(Q, s):
    """
    Greedy policy

    return the index corresponding to the maximum action-state value
    """
    return np.argmax(Q[s])


def run_episodes(env, Q, num_episodes=100, nsteps=100, to_print=False, render=False):
    """
    Run some episodes to test the policy
    """
    print("run_episodes")
    tot_rew = []
    state = env.reset()

    for episode in range(num_episodes):
        print(f"episode={episode}")
        done = False
        game_rew = 0

        for step in range(nsteps):
            # select a greedy action
            next_state, rew, done, truncated, info = env.step(greedy(Q, state))

            state = next_state
            game_rew += rew
            if done:
                state = env.reset()
                tot_rew.append(game_rew)
                break
            if render:
                env.render()

    if to_print:
        print(f"Mean score: {np.mean(tot_rew):.3f} of {num_episodes} games!")

    return np.mean(tot_rew)


def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    print("Q_learning")
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each columns represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        # loop the main body until the environment stops
        while not done:
            # select an action following the eps-greedy policy
            action = eps_greedy(Q, state, eps)

            # Take one step in the environment
            next_state, rew, done, truncated, info = env.step(action)

            # Q-learning update the state-action value (get the max Q value for the next state)
            Q[state][action] = Q[state][action] + lr * (
                    rew + gamma * np.max(Q[next_state]) - Q[state][action]
            )

            state = next_state
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)

        # Test the policy every 300 episodes and print the results
        if (ep % 300) == 0:
            test_rew = run_episodes(env, Q, 10)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)

    return Q


if __name__ == "__main__":
    with gym.make("Taxi-v3", new_step_api=True) as env:
        Q_qlearning = Q_learning(
            env,
            lr=0.1,
            num_episodes=5000,
            eps=0.4,
            gamma=0.95,
            eps_decay=0.001

        )
        env.reset()
        run_episodes(env, Q_qlearning, num_episodes=10, to_print=True, render=True)
