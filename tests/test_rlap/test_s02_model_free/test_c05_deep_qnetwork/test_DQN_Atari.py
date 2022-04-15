import pytest

from rlap.s02_model_free.c05_deep_qnetwork.DQN_Atari import DQN


def test_smoke():
    print("fire?")


@pytest.mark.skip(reason="ModuleNotFoundError: No module named 'gym.envs.atari'")
def test_DQN():
    DQN(
        "PongNoFrameskip-v4",
        hidden_sizes=[128],
        lr=2e-4,
        buffer_size=100000,
        update_target_net=1000,
        batch_size=32,
        update_freq=2,
        frames_num=2,
        min_buffer_size=10000,
        render_cycle=100,
    )
