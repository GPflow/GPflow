

import tensorflow as tf
import numpy as np

def kernel(x1, x2):
    lengthscale = 1  #1.7

    x1s = tf.reduce_sum(tf.square(x1), 1)

    x2s = tf.reduce_sum(tf.square(x2), 1)
    dist = -2 * tf.matmul(x1, x2, transpose_b=True) + \
           tf.reshape(x1s, (-1, 1)) + tf.reshape(x2s, (1, -1))
    return tf.exp(-(dist)/(lengthscale**2))







def k_correct(k, xl, xr, derivs_left, derivs_right):

    output = []
    derivs_left = derivs_left.astype(int)
    derivs_right = derivs_right.astype(int)


    for i in range(int(derivs_left.shape[0])):
        for j in range(int(derivs_right.shape[0])):

            k_new = k[i, j]

            num_derivs_left = derivs_left[i, 0]
            deriv_order_left = derivs_left[i, 1:]
            for _, d in zip(range(num_derivs_left), deriv_order_left):
                k_new = tf.gradients(k_new, xl)[0][i, d]


            num_derivs_right = derivs_right[j, 0]
            deriv_order_right = derivs_right[j, 1:]
            for _, d in zip(range(num_derivs_right), deriv_order_right):
                k_new = tf.gradients(k_new, xr)[0][j, d]

            output.append(k_new)

    full_new_k = tf.stack(output)
    return tf.reshape(full_new_k, tf.shape(k))



def k_correct_dynamic(k, xl, xr, derivs_left, derivs_right):
    k_shape = tf.shape(k)
    k_orig = k

    indcs_x1 = tf.range(0, tf.shape(xl)[0])[:, None] + tf.zeros(tf.shape(k), dtype=tf.int32)
    indcs_x2 = tf.range(0, tf.shape(xr)[0])[None, :] + tf.zeros(tf.shape(k), dtype=tf.int32)

    elems = [tf.reshape(t, (-1,)) for t in ( indcs_x1, indcs_x2)]

    def calc_derivs(tensor_in):

        idxl = tensor_in[0]
        idxr = tensor_in[1]

        k = k_orig[idxl, idxr]


        idx_i = derivs_left[idxl, 1]
        idx_j = derivs_left[idxl, 2]
        idx_k = derivs_right[idxr, 1]
        idx_m = derivs_right[idxr, 2]

        #First order derivatives
        dk__dxli = lambda : tf.gradients(k, xl)[0][idxl, idx_i]
        dk__dxrk = lambda : tf.gradients(k, xr)[0][idxr, idx_k]

        # Second order derivatives
        dk__dxlj_dxli_ = tf.gradients(dk__dxli(), xl)[0][idxl, idx_j]
        dk__dxli_dxrk_ = tf.gradients(dk__dxrk(), xl)[0][idxl, idx_i]
        dk__dxrm_dxrk_ = tf.gradients(dk__dxrk(), xr)[0][idxr, idx_m]
        dk__dxlj_dxli = lambda : dk__dxlj_dxli_
        dk__dxli_dxrk = lambda : dk__dxli_dxrk_
        dk__dxrm_dxrk = lambda : dk__dxrm_dxrk_


        # Third order derivatives
        dk__dxlj_dxli_dxrk = lambda : tf.gradients(dk__dxli_dxrk_, xl)[0][idxl, idx_j]
        dk__dxli_dxrm_dxrk = lambda : tf.gradients(dk__dxrm_dxrk_, xl)[0][idxl, idx_i]

        # Fourth order derivatives
        dk__dxlj_dxli_dxrm_dxrk = lambda : tf.gradients(dk__dxli_dxrm_dxrk(), xl)[0][idxl, idx_j]


        num_left_derivs = derivs_left[idxl, 0]
        num_right_derivs = derivs_right[idxr, 0]
        k_new = tf.case(
            [
                # Zeroth order
                # is done by default
                # First order
                (tf.logical_and(tf.equal(num_left_derivs, 1), tf.equal(num_right_derivs, 0)), dk__dxli),
                (tf.logical_and(tf.equal(num_left_derivs, 0), tf.equal(num_right_derivs, 1)), dk__dxrk),
                # Second order
                (tf.logical_and(tf.equal(num_left_derivs, 2), tf.equal(num_right_derivs, 0)), dk__dxlj_dxli),
                (tf.logical_and(tf.equal(num_left_derivs, 1), tf.equal(num_right_derivs, 1)), dk__dxli_dxrk),
                (tf.logical_and(tf.equal(num_left_derivs, 0), tf.equal(num_right_derivs, 2)), dk__dxrm_dxrk),
                # Third order
                (tf.logical_and(tf.equal(num_left_derivs, 2), tf.equal(num_right_derivs, 1)), dk__dxlj_dxli_dxrk),
                (tf.logical_and(tf.equal(num_left_derivs, 1), tf.equal(num_right_derivs, 2)), dk__dxli_dxrm_dxrk),
                # Fourth order
                (tf.logical_and(tf.equal(num_left_derivs, 2), tf.equal(num_right_derivs, 2)),
                dk__dxlj_dxli_dxrm_dxrk),
            ], default=lambda : k, exclusive=True
        )

        return k_new

    new_kernel = tf.map_fn(calc_derivs, elems, dtype=tf.float64)
    new_kernel_reshaped = tf.reshape(new_kernel, k_shape)
    return new_kernel_reshaped




def main():

    x = np.array([[0, 1.32], [5.2, 6.12], [2.14, 3.21]])
    x_ph = tf.placeholder(tf.float64, [None, None])



    derivs = np.array([[0, -1, -1], [1, 1, -1], [2, 0, 1]])


    print(x_ph)

    x_1 = tf.identity(x_ph)
    x_2 = tf.identity(x_ph)

    x1_actual = x_1
    x2_actual = x_2

    k = kernel(x1_actual, x2_actual)

    k_corrected_wrt_derivs = k_correct(k, x1_actual, x2_actual, derivs, derivs)

    with tf.Session() as sess:
        print("running..")
        sess.run(tf.initialize_all_variables())
        k_ = sess.run(k, feed_dict={x_ph:x})
        print(k_)
        k_cdorrect = sess.run(k_corrected_wrt_derivs, feed_dict={x_ph:x})
        print(k_cdorrect)


def main2():

    x = np.array([[0, 1.32], [5.2, 6.12], [2.14, 3.21]])
    x_ph = tf.placeholder(tf.float64, [None, None])


    derivs_ph = tf.placeholder(tf.int32, [None, 3])

    derivs = np.array([[0, -1, -1], [1, 1, -1], [2, 0, 1]])


    print(x_ph)

    x_1 = tf.identity(x_ph)
    derivs1 = tf.identity(derivs_ph)
    x_2 = tf.identity(x_ph)
    derivs2 = tf.identity(derivs_ph)

    x1_actual = x_1
    x2_actual = x_2

    k = kernel(x1_actual, x2_actual)

    k_corrected_wrt_derivs = k_correct_dynamic(k, x1_actual, x2_actual, derivs1, derivs2)

    with tf.Session() as sess:
        print("running..")
        sess.run(tf.initialize_all_variables())
        k_ = sess.run(k, feed_dict={x_ph:x})
        print(k_)
        print("waiting...")
        k_cdorrect = sess.run(k_corrected_wrt_derivs, feed_dict={x_ph:x, derivs_ph:derivs})
        print(k_cdorrect)









if __name__ == '__main__':
    main2()
    main()