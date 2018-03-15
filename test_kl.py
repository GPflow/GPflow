import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import tensorflow as tf
import numpy as np
from gpflow.kullback_leiblers import  gauss_kl
float_type = tf.float64

M = 20
L= 2

q_mu = tf.constant(np.random.rand(M,L),dtype=float_type)

for K_shape in [None,[M,M],[L,M,M]]:

    if K_shape == None:
        K = None
    elif len(K_shape) == 2:
        K = tf.constant(np.eye(M), dtype=float_type)
    elif len(K_shape) == 3:
        K = tf.constant(np.array([np.eye(M) for _ in range(L)]),dtype=float_type)

    for q_shape in [[M,L],[L,M,M]]:

        if len(q_shape)==2:
            q_sqrt = tf.constant(np.ones(q_shape),dtype=float_type)
        elif len(q_shape)==3:
            q_sqrt = tf.constant(np.array([np.eye(M) for _ in range(L)]),dtype=float_type)

        print(K.shape if K is not None else None, q_sqrt.shape,q_mu.shape)

        KL = gauss_kl(q_mu, q_sqrt, K)
        with tf.Session() as sess:
            print(KL.eval())
