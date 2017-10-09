
import tensorflow as tf


from . import decors

ERF_CODY_LIMIT1 = 0.6629
ERF_CODY_LIMIT2 = 5.6569
M_LN2PI = 1.83787706640934533908193770913
M_LN2 = 0.69314718055994530941723212146
M_SQRTPI = 1.77245385090551602729816748334
M_SQRT2 = 1.41421356237309504880168872421






def log_pdf_normal(z):
    return -0.5*(M_LN2PI+tf.square(z))


def deriv_log_cdf_normal(z, name=None):
    """
    Robust implementations of derivative of the log cdf of a standard normal.
    taken from the C code by Matthias Seeger at
    https://github.com/mseeger/apbsint/blob/master/src/eptools/potentials/SpecfunServices.h

    Also see the GPy project:
    https://github.com/SheffieldML/GPy/blob/devel/GPy/util/univariate_Gaussian.py

    NB only currently works on 1d vectors.
    """
    name = name or "deriv_log_cdf_normal"

    with tf.name_scope(name):
        orig_shape = tf.shape(z)
        z = tf.reshape(z, [-1])

        def inbetween_erf_cody_limit(z):
            # Phi(z) approx (1 + y R_3(y^2))/2, y = z/sqrt(2)
            return tf.identity(2.0 * tf.exp(log_pdf_normal(z)) / (1.0 + (z / M_SQRT2) * _erf_rational_helper_r3(0.5 * tf.square(z))),
                               name="inbetween_erf_cody_limit_op")

        def lower_than_zero(z):
            # Phi(z) approx N(z) Q(-z)/(-z), z<0
            return tf.identity(-z / _erf_rational_helper(-z), name="lower_than_zero_op")

        def default(z):
            t = tf.exp(log_pdf_normal(z), name="pdf_normal_for_def")
            return tf.identity(t / (1.0 - t * _erf_rational_helper(z) / z), name="default_op")


        # So we want to have an elementwise switch function. We could do this in TF with a map and a
        # case. But here instead decided to dynamically partition out into the three different routes
        # And then stitch back the results at the end.

        cases = tf.zeros_like(z, dtype=tf.int32)  # 0 will mean default route
        cases = tf.where(tf.less(tf.abs(z), ERF_CODY_LIMIT1), tf.ones_like(z, dtype=tf.int32), cases)
        # inbetween_erf_cody_limit will be case 1.
        cases = tf.where(tf.less_equal(z, -ERF_CODY_LIMIT1), 2 * tf.ones_like(z, dtype=tf.int32), cases,
                         name="final_cases")
        #^  lower_than_zero will be case 2, but this will only happen if still default and less than zero.

        data_split = tf.dynamic_partition(z, cases, 3)
        partitions = tf.dynamic_partition(tf.range(0, tf.shape(z)[0]), cases, 3)


        default_res = default(data_split[0])
        inbetween_res = inbetween_erf_cody_limit(data_split[1])
        low_than_zero_res = lower_than_zero(data_split[2])

        results_stitched = tf.dynamic_stitch(partitions, [default_res, inbetween_res, low_than_zero_res])

        # res = tf.case([(tf.less(tf.abs(z), ERF_CODY_LIMIT1), inbetween_erf_cody_limit),
        #                (tf.less(z, 0.0), lower_than_zero)],
        #                default=default, exclusive=False)

        results = tf.reshape(results_stitched, orig_shape, name="final_reshaped")
    return results


decors.name_scope("erf_rational_helper")
def _erf_rational_helper(x):
    assertion = tf.assert_positive(x)

    def above_erf_limit(x):
        """
         x/sqrt(2) >= 4
         Q(x)   = 1 + sqrt(pi) y R_1(y),
         R_1(y) = poly(p_j,y) / poly(q_j,y),  where  y = 2/x^2
         Ordering of arrays: 4,3,2,1,0,5 (only for numerator p_j; q_5=1)
         ATTENTION: The p_j are negative of the entries here
        """
        P1_ARRAY = [
            3.05326634961232344e-1, 3.60344899949804439e-1,
            1.25781726111229246e-1, 1.60837851487422766e-2,
            6.58749161529837803e-4, 1.63153871373020978e-2]
        Q1_ARRAY = [
            2.56852019228982242e+0, 1.87295284992346047e+0,
            5.27905102951428412e-1, 6.05183413124413191e-2,
            2.33520497626869185e-3]

        y = 2.0 * tf.reciprocal(tf.square(x))
        res = y * P1_ARRAY[5]
        den = y

        for p1_val, q1_val in zip(P1_ARRAY[:4], Q1_ARRAY[:4]):
            res = (res + p1_val) * y
            den = (den + q1_val) * y

        # Minus, because p(j) values have to be negated
        return tf.identity(1.0 - M_SQRTPI * y * (res + P1_ARRAY[4]) / (den + Q1_ARRAY[4]), name="above_erf_limit_route")

    def else_func(x):
        """
         x/sqrt(2) < 4, x/sqrt(2) >= 0.469
         Q(x)   = sqrt(pi) y R_2(y),
         R_2(y) = poly(p_j,y) / poly(q_j,y),   y = x/sqrt(2)
         Ordering of arrays: 7,6,5,4,3,2,1,0,8 (only p_8; q_8=1)
        """
        P2_ARRAY = [
            5.64188496988670089e-1, 8.88314979438837594e+0,
            6.61191906371416295e+1, 2.98635138197400131e+2,
            8.81952221241769090e+2, 1.71204761263407058e+3,
            2.05107837782607147e+3, 1.23033935479799725e+3,
            2.15311535474403846e-8]
        Q2_ARRAY = [
            1.57449261107098347e+1, 1.17693950891312499e+2,
            5.37181101862009858e+2, 1.62138957456669019e+3,
            3.29079923573345963e+3, 4.36261909014324716e+3,
            3.43936767414372164e+3, 1.23033935480374942e+3]


        y = x / M_SQRT2
        res = y * P2_ARRAY[8]
        den = y

        for p2_val, q2_val in zip(P2_ARRAY[:7], Q2_ARRAY[:7]):
            res = (res + p2_val) * y
            den = (den + q2_val) * y

        return tf.identity(M_SQRTPI * y * (res + P2_ARRAY[7]) / (den + Q2_ARRAY[7]), name="else_route_end")

    # with tf.control_dependencies([assertion]):
    #     result = tf.cond(tf.greater_equal(x, ERF_CODY_LIMIT2),
    #             true_fn=above_erf_limit,
    #             false_fn=else_func)
    with tf.control_dependencies([assertion]):
        cases = tf.where(tf.greater_equal(x, ERF_CODY_LIMIT2), tf.zeros_like(x, dtype=tf.int32), tf.ones_like(x, dtype=tf.int32))
        data_split = tf.dynamic_partition(x, cases, 2)
        partitions = tf.dynamic_partition(tf.range(0, tf.shape(x)[0]), cases, 2)

        abover_erf_lim_res = above_erf_limit(data_split[0])
        else_res = else_func(data_split[1])

        result = tf.dynamic_stitch(partitions, [abover_erf_lim_res, else_res], name="erf_rational_helper_final")
    return result


decors.name_scope("erf_rational_helper_r3")
def _erf_rational_helper_r3(y):
    assertion = tf.assert_non_negative(y)

    P3_ARRAY = [
        3.16112374387056560e+0, 1.13864154151050156e+2,
        3.77485237685302021e+2, 3.20937758913846947e+3,
        1.85777706184603153e-1]
    Q3_ARRAY = [
        2.36012909523441209e+1, 2.44024637934444173e+2,
        1.28261652607737228e+3, 2.84423683343917062e+3]

    nom = y * P3_ARRAY[4]
    den = y

    for p_val, q_val in zip(P3_ARRAY[:3], Q3_ARRAY[:3]):
        nom = (nom + p_val) * y
        den = (den + q_val) * y

    with tf.control_dependencies([assertion]):
        result = (nom + P3_ARRAY[3]) / (den + Q3_ARRAY[3])
    return result
