import numpy as np
import gym
from rlap.s01_algorithms_and_environments.c03_dynamic_programming.frozenlake8x8_valueiteration import (
    value_iteration,
    run_episodes, env
)


def test_smoke():
    print("fire?")


def test_create_env():
    # create the environment
    env = gym.make("FrozenLake-v1")
    # enwrap it to have additional information from it
    env = env.unwrapped


def test_spaces_dimension():
    # create the environment
    env = gym.make("FrozenLake-v1")
    # enwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n


def test_Value_iteration():
    V = value_iteration(eps=0.0001)


def test_run_episodes():
    # Value iteration
    V = value_iteration(eps=0.0001)
    # test the value function on 100 games
    run_episodes(env, V, 100)
    # print the state values
    print(V.reshape((4, 4)))
