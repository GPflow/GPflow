import tensorflow as tf

def jitchol(K, maxtries=5):
    jitter = tf.diag(tf.mul(tf.ones(tf.pack([tf.shape(K)[0],]), dtype='float64'),tf.constant(1e-6, dtype='float64')))
    
    num_tries = 1
    
    while num_tries <= maxtries:
        try:
            L = tf.cholesky(tf.add(K, jitter))
            
            return L
        except:
            jitter = tf.mul(jitter, tf.constant(10, dtype='float64'))
            num_tries += 1

            
    raise Exception("not positive definite, even with jitter.")
    return L