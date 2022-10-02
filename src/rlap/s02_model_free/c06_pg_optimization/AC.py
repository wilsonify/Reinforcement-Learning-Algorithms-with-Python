"""
Actor-Critic Algorithm
"""
import logging

import numpy as np
import tensorflow as tf
import gym
from datetime import datetime
import time


def mlp(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    """
    Multi-layer perceptron
    """
    for l in hidden_layers:
        x = tf.compat.v1.layers.dense(x, units=l, activation=activation)
    return tf.compat.v1.layers.dense(x, units=output_size, activation=last_activation)


def softmax_entropy(logits):
    """
    Softmax Entropy
    """
    return tf.reduce_sum(
        tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1
    )


def discounted_rewards(rews, last_sv, gamma):
    """
    Discounted reward to go

    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value
    """
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma * last_sv
    for i in reversed(range(len(rews) - 1)):
        rtg[i] = rews[i] + gamma * rtg[i + 1]
    return rtg


class Buffer:
    """
    Buffer class to store the experience from a unique policy
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.obs = []
        self.act = []
        self.ret = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        """
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        """
        # store only if the temp_traj list is not empty
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:, 0])
            rtg = discounted_rewards(temp_traj[:, 1], last_sv, self.gamma)
            self.ret.extend(rtg - temp_traj[:, 3])
            self.rtg.extend(rtg)
            self.act.extend(temp_traj[:, 2])

    def get_batch(self):
        return self.obs, self.act, self.ret, self.rtg

    def __len__(self):
        assert len(self.obs) == len(self.act) == len(self.ret) == len(self.rtg)
        return len(self.obs)


def AC(
        env_name,
        hidden_sizes=(32,),
        ac_lr=5e-3,
        cr_lr=8e-3,
        num_epochs=50,
        gamma=0.99,
        steps_per_epoch=100,
        steps_to_print=100,
):
    """
    Actor-Critic Algorithm
    Parameters:
    -----------
    env_name: Name of the environment
    hidden_size: list of the number of hidden units for each layer
    ac_lr: actor learning rate
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    gamma: discount factor
    steps_per_epoch: number of steps per epoch
    """
    tf.compat.v1.reset_default_graph()
    env = gym.make(env_name, new_step_api=True)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n

    logging.info("set Placeholders")
    obs_ph = tf.compat.v1.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name="obs")
    act_ph = tf.compat.v1.placeholder(shape=(None,), dtype=tf.int32, name="act")
    ret_ph = tf.compat.v1.placeholder(shape=(None,), dtype=tf.float32, name="ret")
    rtg_ph = tf.compat.v1.placeholder(shape=(None,), dtype=tf.float32, name="rtg")

    logging.info("COMPUTE THE PG LOSS FUNCTIONS")

    p_logits = mlp(obs_ph, hidden_sizes, act_dim, activation=tf.tanh)  # policy
    act_multn = tf.squeeze(tf.compat.v1.random.multinomial(p_logits, 1))
    actions_mask = tf.one_hot(act_ph, depth=act_dim)
    p_log = tf.reduce_sum(actions_mask * tf.nn.log_softmax(p_logits), axis=1)
    entropy = -tf.reduce_mean(softmax_entropy(p_logits))  # entropy useful to study the algorithms
    p_loss = -tf.reduce_mean(p_log * ret_ph)
    p_opt = tf.compat.v1.train.AdamOptimizer(ac_lr).minimize(p_loss)  # policy optimization

    logging.info("COMPUTE THE VALUE FUNCTION")

    s_values = tf.squeeze(mlp(obs_ph, hidden_sizes, 1, activation=tf.tanh))  # value function
    v_loss = tf.reduce_mean((rtg_ph - s_values) ** 2)  # MSE loss function
    v_opt = tf.compat.v1.train.AdamOptimizer(cr_lr).minimize(v_loss)  # value function optimization

    now = datetime.now()  # Time
    clock_time = f"{now.day}_{now.hour}.{now.minute}.{now.second}"
    print("Time:", clock_time)

    logging.info("Set scalars and histograms for TensorBoard")
    tf.compat.v1.summary.scalar("p_loss", p_loss, collections=["train"])
    tf.compat.v1.summary.scalar("v_loss", v_loss, collections=["train"])
    tf.compat.v1.summary.scalar("entropy", entropy, collections=["train"])
    tf.compat.v1.summary.scalar("s_values", tf.reduce_mean(s_values), collections=["train"])
    tf.compat.v1.summary.histogram("p_soft", tf.nn.softmax(p_logits), collections=["train"])
    tf.compat.v1.summary.histogram("p_log", p_log, collections=["train"])
    tf.compat.v1.summary.histogram("act_multn", act_multn, collections=["train"])
    tf.compat.v1.summary.histogram("p_logits", p_logits, collections=["train"])
    tf.compat.v1.summary.histogram("ret_ph", ret_ph, collections=["train"])
    tf.compat.v1.summary.histogram("rtg_ph", rtg_ph, collections=["train"])
    tf.compat.v1.summary.histogram("s_values", s_values, collections=["train"])
    train_summary = tf.compat.v1.compat.v1.summary.merge_all("train")

    tf.compat.v1.summary.scalar("old_v_loss", v_loss, collections=["pre_train"])
    tf.compat.v1.summary.scalar("old_p_loss", p_loss, collections=["pre_train"])
    pre_scalar_summary = tf.compat.v1.summary.merge_all("pre_train")

    hyp_str = f"steps={steps_per_epoch}, aclr={ac_lr}, crlr={cr_lr}"
    file_writer = tf.compat.v1.compat.v1.summary.FileWriter(
        f"log_dir/{env_name}/AC_{clock_time}_{hyp_str}",
        tf.compat.v1.get_default_graph(),
    )

    logging.info("create a session")
    sess = tf.compat.v1.Session()
    logging.info("initialize the variables")
    sess.run(tf.compat.v1.global_variables_initializer())

    step_count = 0
    train_rewards = []
    train_ep_len = []
    timer = time.time()
    last_print_step = 0

    logging.info("Reset the environment at the beginning of the cycle")
    obs = env.reset()
    ep_rews = []
    logging.info("main cycle")
    for epoch in range(num_epochs):
        logging.debug(f"epoch={epoch}")
        logging.debug(f"initialize buffer and other variables for the new epochs")
        buffer = Buffer(gamma)
        env_buf = []

        logging.debug(f"iterate over a fixed number of iterations")
        for step in range(steps_per_epoch):
            logging.debug("%r", f"step={step}")
            logging.debug("%r", f"run the policy")
            act, val = sess.run([act_multn, s_values], feed_dict={obs_ph: [obs]})
            logging.debug("%r", f"take a step in the environment")
            obs2, rew, done, trunc, _ = env.step(np.squeeze(act))
            logging.debug("%r", f"add the new transition")
            env_buf.append([obs.copy(), rew, act, np.squeeze(val)])
            obs = obs2.copy()
            step_count += 1
            last_print_step += 1
            ep_rews.append(rew)
            if done:
                logging.debug("%r", f"done={done}")
                # store the trajectory just completed
                # Changed from REINFORCE!
                # The second parameter is the estimated value of the next state.
                # Because the environment is done.
                # we pass a value of 0
                buffer.store(np.array(env_buf), 0)
                env_buf = []
                logging.debug("%r", f"store additional information about the episode")
                train_rewards.append(np.sum(ep_rews))
                train_ep_len.append(len(ep_rews))
                logging.debug("%r", f"reset the environment")
                obs = env.reset()
                ep_rews = []

        logging.debug("%r", f"Bootstrap with the estimated state value of the next state!")
        if len(env_buf) > 0:
            last_sv = sess.run(s_values, feed_dict={obs_ph: [obs]})
            buffer.store(np.array(env_buf), last_sv)

        logging.debug("%r", f"collect the episodes' information")
        obs_batch, act_batch, ret_batch, rtg_batch = buffer.get_batch()

        logging.debug("%r", f"run pre_scalar_summary before the optimization phase")
        old_p_loss, old_v_loss, epochs_summary = sess.run(
            [p_loss, v_loss, pre_scalar_summary],
            feed_dict={
                obs_ph: obs_batch,
                act_ph: act_batch,
                ret_ph: ret_batch,
                rtg_ph: rtg_batch,
            },
        )
        file_writer.add_summary(epochs_summary, step_count)

        logging.debug("%r", f"Optimize the actor and the critic")
        sess.run(
            [p_opt, v_opt],
            feed_dict={
                obs_ph: obs_batch,
                act_ph: act_batch,
                ret_ph: ret_batch,
                rtg_ph: rtg_batch,
            },
        )

        logging.debug("%r", f"run train_summary to save the summary after the optimization")
        new_p_loss, new_v_loss, train_summary_run = sess.run(
            [p_loss, v_loss, train_summary],
            feed_dict={
                obs_ph: obs_batch,
                act_ph: act_batch,
                ret_ph: ret_batch,
                rtg_ph: rtg_batch,
            },
        )
        file_writer.add_summary(train_summary_run, step_count)
        summary = tf.compat.v1.Summary()
        summary.value.add(tag="diff/p_loss", simple_value=(old_p_loss - new_p_loss))
        summary.value.add(tag="diff/v_loss", simple_value=(old_v_loss - new_v_loss))
        file_writer.add_summary(summary, step_count)
        file_writer.flush()

        logging.debug("%r", f"it's time to print some useful information")
        if last_print_step > steps_to_print:
            print(
                f"""
                Ep:{epoch} 
                MnRew:{np.mean(train_rewards)} 
                MxRew:{np.max(train_rewards)} 
                EpLen:{np.mean(train_ep_len)} 
                Buffer:{len(buffer)} -- 
                Step:{step_count} -- 
                Time:{time.time() - timer}
                """
            )

            summary = tf.compat.v1.Summary()
            summary.value.add(tag="supplementary/len", simple_value=np.mean(train_ep_len))
            summary.value.add(tag="supplementary/train_rew", simple_value=np.mean(train_rewards))
            file_writer.add_summary(summary, step_count)
            file_writer.flush()
            timer = time.time()
            train_rewards = []
            train_ep_len = []
            last_print_step = 0

    env.close()
    file_writer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.disable_eager_execution()
    AC(
        env_name="LunarLander-v2",
        hidden_sizes=[64],
        ac_lr=4e-3,
        cr_lr=1.5e-2,
        gamma=0.99,
        steps_per_epoch=100,
        steps_to_print=5000,
        num_epochs=8000,
    )
