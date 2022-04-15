import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
import time

from rlap.s02_model_free.c06_pg_optimization.AC import AC


def test_smoke():
    print("fire?")


def test_AC():
    AC(
        "LunarLander-v2",
        hidden_sizes=[64],
        ac_lr=4e-3,
        cr_lr=1.5e-2,
        gamma=0.99,
        steps_per_epoch=100,
        steps_to_print=5000,
        num_epochs=8000,
    )
