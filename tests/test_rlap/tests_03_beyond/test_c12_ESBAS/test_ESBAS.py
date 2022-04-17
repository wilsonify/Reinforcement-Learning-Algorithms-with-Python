import numpy as np
import tensorflow as tf
import gym
from datetime import datetime

from rlap.s03_beyond.c12_ESBAS.ESBAS import (
    current_milli_time,
    DQN_optimization,
    ExperienceBuffer,
    UCB1,
    eps_greedy,
    test_agent
)


def test_smoke():
    print("fire?")


def test_ESBAS():
    env_name = "Acrobot-v1"
    discount = 0.99
    update_target_net = 1000
    batch_size = 64
    update_freq = 4
    min_buffer_size = 5000
    start_explor = 1
    hidden_sizes = [[64], [16, 16], [64, 64]]
    lr = 4e-4
    buffer_size = 100000
    update_target_net = 100
    batch_size = 32
    update_freq = 4
    min_buffer_size = 100
    explor_steps = 50000
    num_epochs = 20000
    end_explor = 0.1
    xi = 1.0 / 4

    # reset the default graph
    tf.compat.v1.reset_default_graph()

    # Create the environment both for train and test
    env = gym.make(env_name)
    # Add a monitor to the test env to store the videos
    env_test = gym.wrappers.RecordVideo(
        env=gym.make(env_name),
        video_folder="VIDEOS/TEST_VIDEOS" + env_name + str(current_milli_time()),
        video_length=100,
        step_trigger=lambda x: x % 20 == 0,
    )

    dqns = []
    for l in hidden_sizes:
        dqns.append(
            DQN_optimization(
                env.observation_space.shape, env.action_space.n, l, lr, discount
            )
        )

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print("Time:", clock_time)

    LOG_DIR = "log_dir/" + env_name
    hyp_str = "-lr_{}-upTN_{}-upF_{}-xi_{}".format(
        lr, update_target_net, update_freq, xi
    )

    # initialize the File Writer for writing TensorBoard summaries
    file_writer = tf.compat.v1.summary.FileWriter(
        LOG_DIR + "/ESBAS_" + clock_time + "_" + hyp_str, tf.compat.v1.get_default_graph()
    )

    def DQNs_update(step_counter):
        # If it's time to train the network:
        if len(buffer) > min_buffer_size and (step_counter % update_freq == 0):

            # sample a minibatch from the buffer
            mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(
                batch_size
            )

            for dqn in dqns:
                dqn.optimize(mb_obs, mb_rew, mb_act, mb_obs2, mb_done)

        # Every update_target_net steps, update the target network
        if len(buffer) > min_buffer_size and (step_counter % update_target_net == 0):

            for dqn in dqns:
                dqn.update_target_network()

    step_count = 0
    episode = 0
    beta = 1

    # Initialize the experience buffer
    buffer = ExperienceBuffer(buffer_size)

    obs = env.reset()

    # policy exploration initialization
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps

    for ep in range(num_epochs):

        # Policies' training
        for i in range(2 ** (beta - 1), 2 ** beta):
            DQNs_update(i)

        ucb1 = UCB1(dqns, xi)
        list_bests = []
        ep_rew = []
        beta += 1

        while step_count < 2 ** beta:

            # Chose the best policy's algortihm that will run the next trajectory
            best_dqn = ucb1.choose_algorithm()
            list_bests.append(best_dqn)

            summary = tf.compat.v1.Summary()
            summary.value.add(tag="algorithm_selected", simple_value=best_dqn)
            file_writer.add_summary(summary, step_count)
            file_writer.flush()

            g_rew = 0
            done = False

            while not done:
                # Epsilon decay
                if eps > end_explor:
                    eps -= eps_decay

                # Choose an eps-greedy action
                act = eps_greedy(np.squeeze(dqns[best_dqn].act(obs)), eps=eps)

                # execute the action in the environment
                obs2, rew, done, _ = env.step(act)

                # Add the transition to the replay buffer
                buffer.add(obs, rew, act, obs2, done)

                obs = obs2
                g_rew += rew
                step_count += 1

            # Update the UCB parameters of the algortihm just used
            ucb1.update(best_dqn, g_rew)

            # The environment is ended.. reset it and initialize the variables
            obs = env.reset()
            ep_rew.append(g_rew)
            g_rew = 0
            episode += 1

            # Print some stats and test the best policy
            summary = tf.compat.v1.Summary()
            summary.value.add(tag="train_performance", simple_value=np.mean(ep_rew))

            if episode % 10 == 0:
                unique, counts = np.unique(list_bests, return_counts=True)
                print(dict(zip(unique, counts)))

                test_agent_results = test_agent(
                    env_test, dqns[best_dqn].act, num_games=10, summary=summary
                )
                print(
                    "Epoch:%4d Episode:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f Best:%2d Last:%2d"
                    % (
                        ep,
                        episode,
                        np.mean(ep_rew),
                        eps,
                        step_count,
                        np.mean(test_agent_results),
                        best_dqn,
                        g_rew,
                    )
                )

            file_writer.add_summary(summary, step_count)
            file_writer.flush()

    file_writer.close()
    env.close()
