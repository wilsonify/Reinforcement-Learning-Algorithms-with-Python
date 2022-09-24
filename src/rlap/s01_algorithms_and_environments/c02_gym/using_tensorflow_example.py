"""
TensorFlow is a machine learning framework that performs high-performance numerical computations.
TensorFlow owes its popularity to its high quality and vast amount of documentation,
its ability to easily serve models at scale in production environments,
and the friendly interface to GPUs and TPUs.
TensorFlow, to facilitate the development and deployment of ML models, has many high-level APIs,
including Keras, Eager Execution, and Estimators. These APIs are very useful in many contexts,
 but, in order to develop RL algorithms, we'll only use low-level APIs.
"""
import logging

import numpy as np
import tensorflow as tf
from datetime import datetime


def tf_example01():
    # create a session

    with tf.compat.v1.Session() as session:
        # create two constants: a and b
        a = tf.constant(4)
        b = tf.constant(3)

        # perform a computation
        c = a + b
        print(c)  # print the shape of c

        # run the session. It computes the sum
        res = session.run(c)
        print(res)  # print the actual result


def tf_example02():
    with tf.compat.v1.Session() as session:
        # ### Tensor
        a = tf.constant(1)
        print(a.shape)

        # array of five elements
        b = tf.constant([1, 2, 3, 4, 5])
        print(b.shape)

        # NB: a can be of any type of tensor
        a = tf.constant([1, 2, 3, 4, 5])
        first_three_elem = a[:3]
        fourth_elem = a[3]

        print(session.run(first_three_elem))
        print(session.run(fourth_elem))


def tf_example03():
    with tf.compat.v1.Session() as session:
        # #### Constant
        a = tf.constant([1.0, 1.1, 2.1, 3.1], dtype=tf.float32, name="a_const")
        print(a)

        # #### Placeholder
        a = tf.compat.v1.placeholder(shape=(1, 3), dtype=tf.float32)
        b = tf.constant([[10, 10, 10]], dtype=tf.float32)
        c = a + b

        res = session.run(c, feed_dict={a: [[0.1, 0.2, 0.3]]})
        print(res)


def tf_example04():
    with tf.compat.v1.Session() as session:
        # NB: the fist dimension is 'None', meaning that it can be of any lenght
        a = tf.compat.v1.placeholder(shape=(None, 3), dtype=tf.float32)
        b = tf.compat.v1.placeholder(shape=(None, 3), dtype=tf.float32)

        c = a + b

        print(a)

        print(session.run(c, feed_dict={a: [[0.1, 0.2, 0.3]], b: [[10, 10, 10]]}))

        v_a = np.array([[1, 2, 3], [4, 5, 6]])
        v_b = np.array([[6, 5, 4], [3, 2, 1]])

        print(session.run(c, feed_dict={a: v_a, b: v_b}))
        print(session.run(c, feed_dict={a: [[0.1, 0.2, 0.3]], b: [[10, 10, 10]]}))


def tf_example05():
    with tf.compat.v1.Session() as session:  # reset the graph
        # #### Variable
        # variable initialized using the glorot uniform initializer
        var = tf.compat.v1.get_variable(
            "first_variable",
            shape=[1, 3],
            dtype=tf.float32,
            initializer=tf.compat.v1.glorot_uniform_initializer,
        )

        # variable initialized with constant values
        init_val = np.array([4, 5])
        var2 = tf.compat.v1.get_variable(
            "second_variable",
            shape=[1, 2],
            dtype=tf.int32,
            initializer=tf.constant_initializer(init_val),
        )

        # initialize all the variables
        session.run(tf.compat.v1.global_variables_initializer())

        print(session.run(var))
        print(session.run(var2))

        # not trainable variable
        var2 = tf.compat.v1.get_variable(
            "variable", shape=[1, 2], trainable=False, dtype=tf.int32
        )
        print(tf.compat.v1.global_variables())


def tf_example06():
    with tf.compat.v1.Session() as session:
        const1 = tf.compat.v1.constant(3.0, name="constant1")
        var = tf.compat.v1.get_variable("variable1", shape=[1, 2], dtype=tf.float32)
        var2 = tf.compat.v1.get_variable(
            "variable2", shape=[1, 2], trainable=False, dtype=tf.float32
        )

        op1 = const1 * var
        op2 = op1 + var2
        op3 = tf.reduce_mean(op2)

        session.run(tf.compat.v1.global_variables_initializer())
        session.run(op3)

        # ### Simple Linear Regression Example


def tf_example07():
    with tf.compat.v1.Session() as session:

        np.random.seed(10)
        tf.compat.v1.set_random_seed(10)

        W, b = 0.5, 1.4

        # create a dataset of 100 examples
        X = np.linspace(0, 100, num=100)

        # add random noise to the y labels
        y = np.random.normal(loc=W * X + b, scale=2.0, size=len(X))

        # create the placeholders
        x_ph = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.float32, )
        y_ph = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.float32, )

        # create the variables.
        v_weight = tf.compat.v1.get_variable("weight", shape=[1], dtype=tf.float32)
        v_bias = tf.compat.v1.get_variable("bias", shape=[1], dtype=tf.float32)

        # linear computation
        out = v_weight * x_ph + v_bias

        # compute the Mean Squared Error
        loss = tf.reduce_mean((out - y_ph) ** 2)

        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(0.4).minimize(loss)

        session.run(tf.compat.v1.global_variables_initializer())

        # loop to train the parameters
        for ep in range(210):
            # run the optimizer and get the loss
            train_loss, _ = session.run([loss, opt], feed_dict={x_ph: X, y_ph: y})
            # print epoch number and loss
            msg = f"Epoch: {ep}, MSE: {train_loss}, W: {session.run(v_weight)}, b: {session.run(v_bias)}"
            if ep % 40 == 0:
                print(msg)

        print("Final weight: %.3f, bias: %.3f" % (session.run(v_weight), session.run(v_bias)))


def tf_example08():
    """with TensorBoard"""

    with tf.compat.v1.Session() as session:
        np.random.seed(10)
        tf.compat.v1.set_random_seed(10)

        W, b = 0.5, 1.4

        # create a dataset of 100 examples
        X = np.linspace(0, 100, num=100)

        # add random noise to the y labels
        y = np.random.normal(loc=W * X + b, scale=2.0, size=len(X))

        # create the placeholders
        x_ph = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.float32)
        y_ph = tf.compat.v1.placeholder(shape=[None, ], dtype=tf.float32)

        # create the variables.
        v_weight = tf.compat.v1.get_variable("weight_of_line", shape=[1], dtype=tf.float32)
        v_bias = tf.compat.v1.get_variable("bias_of_line", shape=[1], dtype=tf.float32)

        # linear computation
        out = v_weight * x_ph + v_bias

        # compute the Mean Squared Error
        loss = tf.reduce_mean((out - y_ph) ** 2)

        # optimizer
        opt = tf.compat.v1.train.AdamOptimizer(0.4).minimize(loss)

        tf.summary.scalar("MSEloss", loss)
        tf.summary.histogram("model_weight", v_weight)
        tf.summary.histogram("model_bias", v_bias)

        now = datetime.now()
        clock_time = f"{now.day}_{now.hour}.{now.minute}.{now.second}"
        with tf.compat.v1.summary.FileWriter("log_dir/" + clock_time, tf.compat.v1.get_default_graph()) as file_writer:
            session.run(tf.compat.v1.global_variables_initializer())
            # loop to train the parameters
            for ep in range(210):
                # run the optimizer and get the loss
                train_loss, train_opt = session.run([loss, opt], feed_dict={x_ph: X, y_ph: y})
                # print epoch number and loss
                if ep % 40 == 0:
                    msg = f"Epoch: {ep}, MSE: {train_loss}, W: {session.run(v_weight)}, b: {session.run(v_bias)}"
                    print(msg)
            print(f"Final weight: {session.run(v_weight)}, bias: {session.run(v_bias)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tf.compat.v1.disable_eager_execution()
    logging.info("start tf_example01")
    tf_example01()
    logging.info("done tf_example01")

    logging.info("start tf_example02")
    tf_example02()
    logging.info("done tf_example02")

    logging.info("start tf_example03")
    tf_example03()
    logging.info("done tf_example03")

    logging.info("start tf_example04")
    tf_example04()
    logging.info("done tf_example04")

    logging.info("start tf_example05")
    tf_example05()
    logging.info("done tf_example05")

    logging.info("start tf_example06")
    tf_example06()
    logging.info("done tf_example06")

    logging.info("start tf_example07")
    tf_example07()
    logging.info("done tf_example07")

    logging.info("start tf_example08")
    tf_example08()
    logging.info("done tf_example08")
