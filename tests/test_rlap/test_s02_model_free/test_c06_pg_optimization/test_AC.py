import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

import gym
from datetime import datetime
import time

from rlap.s02_model_free.c06_pg_optimization.AC import AC


def test_smoke():
    print("fire?")


def test_AC():
    disable_eager_execution()
    AC(
        "LunarLander-v2",
        hidden_sizes=[64],
        ac_lr=4e-3,
        cr_lr=1.5e-2,
        gamma=0.99,
        steps_per_epoch=100,
        steps_to_print=500,
        num_epochs=100,
    )
