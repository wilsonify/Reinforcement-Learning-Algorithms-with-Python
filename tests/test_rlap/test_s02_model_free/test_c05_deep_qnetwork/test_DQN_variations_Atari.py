import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
from collections import deque
import time
import sys

from rlap.s02_model_free.c05_deep_qnetwork.DQN_variations_Atari import DQN_with_variations


def test_smoke():
    print("fire?")


def test_DQN_with_variations():
    extensions_hyp = {"DDQN": False, "dueling": False, "multi_step": 1}
    DQN_with_variations(
        "PongNoFrameskip-v4",
        extensions_hyp,
        hidden_sizes=[128],
        lr=2e-4,
        buffer_size=100000,
        update_target_net=1000,
        batch_size=32,
        update_freq=2,
        frames_num=2,
        min_buffer_size=10000,
        render_cycle=10000,
    )
