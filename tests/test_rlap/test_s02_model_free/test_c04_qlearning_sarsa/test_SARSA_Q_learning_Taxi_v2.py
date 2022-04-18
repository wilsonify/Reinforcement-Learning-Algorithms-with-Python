import numpy as np
import gym
from rlap.s02_model_free.c04_qlearning_sarsa.q_learning_taxi import Q_learning
from rlap.s02_model_free.c04_qlearning_sarsa.sarsa_taxi import SARSA


def test_smoke():
    print("fire?")


def test_Q_learning():
    env = gym.make("Taxi-v3")

    Q_qlearning = Q_learning(
        env, lr=0.1, num_episodes=500, eps=0.4, gamma=0.95, eps_decay=0.001
    )


def test_sarsa():
    env = gym.make("Taxi-v3")
    Q_sarsa = SARSA(
        env, lr=0.1, num_episodes=500, eps=0.4, gamma=0.95, eps_decay=0.001
    )
