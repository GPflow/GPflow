

import tensorflow as tf
import numpy as np

def kernel(x1, x2):
    lengthscale = 1  #1.7

    x1s = tf.reduce_sum(tf.square(x1), 1)

    x2s = tf.reduce_sum(tf.square(x2), 1)
    dist = -2 * tf.matmul(x1, x2, transpose_b=True) + \
           tf.reshape(x1s, (-1, 1)) + tf.reshape(x2s, (1, -1))
    return tf.exp(-(dist)/(lengthscale**2))


def main_parallel():

    x = tf.Variable(np.arange(10)[:, None], dtype=tf.float32)
    x2 = tf.Variable(np.arange(10)[:, None], dtype=tf.float32)

    k = kernel(x, x2)[0,0]

    grad1 = tf.gradients(k, x)
    grad2 = tf.gradients(grad1, x2)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        kern_out = sess.run(grad2)

    print(kern_out)


def one_d_kernel(x1, x2):
    return tf.exp(-(x1-x2)**2)

def main():

    x = tf.placeholder(tf.float32, shape=[1])
    x2 = tf.placeholder(tf.float32, shape=[1])

    k = one_d_kernel(x, x2)

    g1 = tf.gradients(k, x)
    g2 = tf.gradients(g1, x2)


    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        kern_out = sess.run(g2, feed_dict={x:[1], x2:[1]})
        print(kern_out)
        kern_out = sess.run(g2, feed_dict={x:[2], x2:[2]})
        print(kern_out)
        kern_out = sess.run(g2, feed_dict={x:[2], x2:[5]})
        print(kern_out)

        print(np.exp(-3**2)*3**2*(-4)+2* np.exp(-3**2))






if __name__ == '__main__':
    main()