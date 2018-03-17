import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from prettytable import PrettyTable

import tensorflow as tf
import numpy as np
from gpflow.kullback_leiblers import  gauss_kl
float_type = tf.float64
np.random.seed(10)
M = 4
L= 3


# Expected difference white / not white

# KL(N0||N1) = .5* { tr(S1^-1 S0) + m0^T S1^-1 m0 -k + ln ( |S1|) - ln(|S0|)  }
# KL(N0||Id) = .5* { tr(S0) + m0^T m0 -k  - ln ( |S0|)  }

# KL(N0||N1) - KL(N0||Id)=
# .5* { tr(S1^-1 S0) + (m0)^T S1^-1 (m0) -k + ln ( |S1|) - ln(|S0|)  }
# - .5* { tr(S0) + m0^T m0 -k  - ln ( |S0|)  }
# = .5 * { tr([S1^-1 - Id] S0) + (m0)^T [S1^-1 - Id] (m0)  + ln ( |S1|) }



A = np.random.randn(M,M)/M
A = (A+A.T)
B = np.random.randn(M,M)/M
B = (B+B.T)

q_mu = tf.constant(np.random.rand(M,L),dtype=float_type)


q_sqrt = tf.constant(np.ones([M, L]), dtype=float_type)




K = tf.constant(np.eye(M) , dtype=float_type)



KL = gauss_kl(q_mu, q_sqrt, K)

with tf.Session() as sess:
    print(gauss_kl(q_mu, q_sqrt, K).eval())
    print(gauss_kl(q_mu, q_sqrt, None).eval())

print('=======================')

t = PrettyTable([' p(x)', 'q_sqrt','q_mu', 'value'])



for q_shape in [[M, L], [L, M, M]]:

    if len(q_shape) == 2:
        q_sqrt = tf.constant(np.ones(q_shape), dtype=float_type)
    elif len(q_shape) == 3:
        q_sqrt = tf.constant(np.array([np.eye(M)+B for _ in range(L)]), dtype=float_type)

#    for K_shape in [None,[M,M],[L,M,M]]:
    for K_shape in [ [M, M], [L, M, M]]:
        # None means white.

        if K_shape == None:
            K = None
        elif len(K_shape) == 2:
            K = tf.constant(np.eye(M)+A, dtype=float_type)
        elif len(K_shape) == 3:
            K = tf.constant(np.array([np.eye(M)+A for _ in range(L)]),dtype=float_type)


        white = True if K is None else False


        KL = gauss_kl(q_mu, q_sqrt, K)
        with tf.Session() as sess:
            KL_np = KL.eval()

        t.add_row([
            'white' if white else 'not white %s '%str(K.shape), q_sqrt.shape, q_mu.shape, KL_np
        ])

print(t)



print('======================= diff white / not white , shared q_sqrt shape')


t = PrettyTable(['K (non eye)', 'q_sqrt','q_mu', 'KL - KL_white'])




for q_shape in [[M, L], [L, M, M]]:

    if len(q_shape) == 2:
        q_sqrt = tf.constant(np.ones(q_shape), dtype=float_type)
    elif len(q_shape) == 3:
        q_sqrt = tf.constant(np.array([np.eye(M)+B for _ in range(L)]), dtype=float_type)

    KL_white  = gauss_kl(q_mu, q_sqrt, None)


    for K_shape in [[M,M],[L,M,M]]:
        # None means white.

        if K_shape == None:
            K = None
        elif len(K_shape) == 2:
            K = tf.constant(np.eye(M)+A, dtype=float_type)
        elif len(K_shape) == 3:
            K = tf.constant(np.array([np.eye(M)+A for _ in range(L)]),dtype=float_type)


        white = True if K is None else False

        KL = gauss_kl(q_mu, q_sqrt, K)

        with tf.Session() as sess:
            diff_np = (KL - KL_white).eval()

        t.add_row(
            [K.shape, q_sqrt.shape, q_mu.shape, diff_np]
        )

print(t)

