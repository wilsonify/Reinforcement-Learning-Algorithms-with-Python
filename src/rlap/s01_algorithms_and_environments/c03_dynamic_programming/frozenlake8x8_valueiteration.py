"""

"""

from pprint import pprint

import numpy as np
import gym


def eval_state_action(env, V, s, a, gamma=0.99):
    """
    update using the Bellman equation
    """
    return np.sum([p * (rew + gamma * V[next_s]) for p, next_s, rew, _ in env.P[s][a]])


def value_iteration(env, nA, nS, eps=0.0001):
    """
    Value iteration algorithm
    Value iteration is another dynamic programming algorithm to find optimal values in an MDP.
    value iteration combines policy evaluations and policy iterations in a single update.

    it updates the value of a state by selecting the best action immediately:
    The code for value iteration is simpler than the policy iteration

    The only difference is in the new value estimation update and in the absence of a proper
    policy iteration module.
    """
    V = np.zeros(nS)
    it = 0

    while True:
        delta = 0
        # update the value of each state using as "policy" the max operator
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(env, V, s, a) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break
        else:
            print(f"Iter:{it} delta:{np.round(delta, 5)}")
        it += 1

    return V


def run_episodes(env, nA, V, num_games=100):
    """
    Run some test games
    """
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            action = np.argmax([eval_state_action(env, V, state, a) for a in range(nA)])
            next_state, reward, done, truncated, info = env.step(action)

            state = next_state
            tot_rew += reward
            if done:
                state = env.reset()

    print("Won %i of %i games!" % (tot_rew, num_games))


def main():
    # create the environment
    env = gym.make("FrozenLake-v1", new_step_api=True)
    # enwrap it to have additional information from it
    env = env.unwrapped
    pprint(env)
    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n
    # Value iteration
    V = value_iteration(env, nA, nS, eps=0.0001)
    # test the value function on 100 games
    run_episodes(env, nA, V, 100)
    # print the state values
    print(V.reshape((4, 4)))


if __name__ == "__main__":
    main()
