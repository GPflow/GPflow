import itertools

import gpflow
import gpflow.actions
import gpflow.training.monitor as mon
from gpflow.test_util import session_tf
import numpy as np
import tensorflow as tf


def test_monitor(session_tf):
    np.random.seed(0)
    X = np.random.rand(10000, 1) * 10
    Y = np.sin(X) + np.random.randn(*X.shape)

    with gpflow.defer_build():
        m = gpflow.models.SVGP(X, Y, gpflow.kernels.RBF(1), gpflow.likelihoods.Gaussian(),
                               Z=np.linspace(0, 10, 5)[:, None],
                               minibatch_size=100, name="SVGP")
        m.likelihood.variance = 0.01
    m.compile()

    global_step = tf.Variable(0, trainable=False, name="global_step")
    session_tf.run(global_step.initializer)

    adam = gpflow.train.AdamOptimizer(0.01).make_optimize_action(m, global_step=global_step)

    # create a filewriter for summaries
    fw = tf.summary.FileWriter('./model_tensorboard', m.graph)

    print_lml = mon.PrintTimings(itertools.count(), mon.Trigger.ITER, single_line=True, global_step=global_step)
    sleep = mon.SleepAction(itertools.count(), mon.Trigger.ITER, 0.0)
    saver = mon.StoreSession(itertools.count(step=3), mon.Trigger.ITER, session_tf,
                             hist_path="./monitor-saves/checkpoint", global_step=global_step)
    tensorboard = mon.ModelTensorBoard(itertools.count(step=3), mon.Trigger.ITER, m, fw, global_step=global_step)
    lml_tensorboard = mon.LmlTensorBoard(itertools.count(step=5), mon.Trigger.ITER, m, fw, global_step=global_step)
    callback = mon.CallbackAction(mon.seq_exp_lin(2.0, np.inf, 1e-3), mon.Trigger.TOTAL_TIME, lambda x, b: x, m)

    actions = [adam, print_lml, tensorboard, lml_tensorboard, saver, sleep, callback]

    gpflow.actions.Loop(actions, stop=11)()
