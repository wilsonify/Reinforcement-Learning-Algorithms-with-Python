from pprint import pprint

import numpy as np
import gym
from rlap.s01_algorithms_and_environments.c03_dynamic_programming.frozenlake8x8_policyiteration import (
    policy_evaluation,
    policy_improvement,
    run_episodes
)


def test_smoke():
    print("fire?")


def test_create_environment():
    env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    pprint(env)


def test_spaces_dimension():
    env = gym.make("FrozenLake-v1")
    # enwrap it to have additional information from it
    env = env.unwrapped
    nA = env.action_space.n
    nS = env.observation_space.n


def test_initializing():
    env = gym.make("FrozenLake-v1")
    env = env.unwrapped
    nA = env.action_space.n
    nS = env.observation_space.n
    V = np.zeros(nS)
    policy = np.zeros(nS)


def test_policy_evaluation():
    # create the environment
    env = gym.make("FrozenLake-v1")
    # enwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n

    # initializing value function and policy
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # some useful variable
    policy_stable = False
    it = 0

    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)
        it += 1


def test_run_episodes():
    # create the environment
    env = gym.make("FrozenLake-v1")
    # enwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n

    # initializing value function and policy
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # some useful variable
    policy_stable = False
    it = 0

    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)
        it += 1

    print("Converged after %i policy iterations" % (it))
    run_episodes(env, policy)
    print(V.reshape((4, 4)))
    print(policy.reshape((4, 4)))
