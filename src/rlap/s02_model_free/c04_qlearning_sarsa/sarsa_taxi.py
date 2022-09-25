"""
The name SARSA comes from the update that is based on
state
action
reward
state,
action,
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

    for _ in range(num_episodes):
        done = False
        game_rew = 0

        for step in range(nsteps):
            print(f"step={step}")
            # select a greedy action
            next_state, rew, done, trunc, _ = env.step(greedy(Q, state))

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


def SARSA(env, lr=0.01, num_episodes=10000, nsteps=100, eps=0.3, gamma=0.95, eps_decay=0.00005):
    print("SARSA")
    nA = env.action_space.n
    nS = env.observation_space.n

    # Initialize the Q matrix
    # Q: matrix nS*nA where each row represent a state and each colums represent a different action
    Q = np.zeros((nS, nA))
    games_reward = []
    test_rewards = []

    for ep in range(1, num_episodes):
        state = env.reset()
        done = False
        tot_rew = 0

        # decay the epsilon value until it reaches the threshold of 0.01
        if eps > 0.01:
            eps -= eps_decay

        action = eps_greedy(Q, state, eps)

        # loop the main body until the environment stops
        for step in range(nsteps):
            next_state, rew, done, trunc, _ = env.step(
                action
            )  # Take one step in the environment

            # choose the next action (needed for the SARSA update)
            next_action = eps_greedy(Q, next_state, eps)
            # SARSA update
            Q[state][action] = Q[state][action] + lr * (
                    rew + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state = next_state
            action = next_action
            tot_rew += rew
            if done:
                games_reward.append(tot_rew)
                state = env.reset()
                break

        # Test the policy every 300 episodes and print the results
        if (ep % 300) == 0 and ep > 0:
            test_rew = run_episodes(env, Q, 1)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_rew))
            test_rewards.append(test_rew)

    return Q


if __name__ == "__main__":
    with gym.make("Taxi-v3", new_step_api=True) as env:
        Q_sarsa = SARSA(
            env=env,
            lr=0.1,
            num_episodes=5000,
            eps=0.4,
            gamma=0.95,
            eps_decay=0.001
        )
        run_episodes(env, Q_sarsa, 5, to_print=True, render=True)
