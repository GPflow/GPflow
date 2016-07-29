import tensorflow as tf
from GPflow.tfops import vec_to_tri

with tf.Session(''):
    print(vec_to_tri([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]).eval())
    print("")
    # print(vec_to_tri([[0, 1, 2, 3]]).eval())
    # print("")
    
