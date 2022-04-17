import tensorflow as tf
from datetime import datetime

import multiprocessing as mp
import numpy as np

from rlap.s03_beyond.c11_black_box_optimization.ES import worker, normalized_rank


def test_smoke():
    print("fire?")


def test_ES():
    env_name = "LunarLanderContinuous-v2"

    hidden_sizes = [32, 32]
    number_iter = 200
    num_workers = 4
    lr = 0.02
    indiv_per_worker = 12
    std_noise = 0.05

    initial_seed = np.random.randint(1e7)

    # Create a queue for the output values (single returns and seeds values)
    output_queue = mp.Queue(maxsize=num_workers * indiv_per_worker)
    # Create a queue for the input paramaters (batch return and batch seeds)
    params_queue = mp.Queue(maxsize=num_workers)

    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    hyp_str = "-numworkers_" + str(num_workers) + "-lr_" + str(lr)
    file_writer = tf.compat.v1.summary.FileWriter(
        "log_dir/" + env_name + "/" + clock_time + "_" + hyp_str, tf.compat.v1.get_default_graph()
    )

    processes = []
    # Create a parallel process for each worker
    for widx in range(num_workers):
        p = mp.Process(
            target=worker,
            args=(
                env_name,
                initial_seed,
                hidden_sizes,
                lr,
                std_noise,
                indiv_per_worker,
                str(widx),
                params_queue,
                output_queue,
            ),
        )
        p.start()
        processes.append(p)

    tot_steps = 0
    # Iterate over all the training iterations
    for n_iter in range(number_iter):

        batch_seed = []
        batch_return = []

        # Wait until enough candidate individuals are evaluated
        for _ in range(num_workers * indiv_per_worker):
            p_rews, p_seed, p_steps = output_queue.get()

            batch_seed.append(p_seed)
            batch_return.extend(p_rews)
            tot_steps += p_steps

        print("Iter: {} Reward: {:.2f}".format(n_iter, np.mean(batch_return)))

        # Let's save the population's performance
        summary = tf.compat.v1.Summary()
        for r in batch_return:
            summary.value.add(tag="performance", simple_value=r)
        file_writer.add_summary(summary, tot_steps)
        file_writer.flush()

        # Rank and normalize the returns
        batch_return = normalized_rank(batch_return)

        # Put on the queue all the returns and seed so that each worker can optimize the neural network
        for _ in range(num_workers):
            params_queue.put([batch_return, batch_seed])

    # terminate all workers
    for p in processes:
        p.terminate()
