import tensorflow as tf

from . import optimizer


class NatGradOptimizer(optimizer.Optimizer):
    """
    """

    def __init__(self, gamma):
        self._gamma = gamma
    
    @property
    def gamma(self):
        """
        """
        return self.gamma

    def minimize(self, model, session=None, var_list=None, feed_dict=None,
                 maxiter=1000, initialize=False, anchor=True, **kwargs):
        """
        Minimizes objective function of the model.

        :param model: GPflow model with objective tensor.
        :param session: Session where optimization will be run.
        :param var_list: List of extra variables which should be trained during optimization.
        :param feed_dict: Feed dictionary of tensors passed to session run method.
        :param maxiter: Number of run interation.
        :param initialize: If `True` model parameters will be re-initialized even if they were
            initialized before for gotten session.
        :param anchor: If `True` trained variable values computed during optimization at
            particular session will be synchronized with internal parameter values.
        :param kwargs: This is a dictionary of extra parameters for session run method.
        """

        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        session = model.enquire_session(session)

        self._model = model
        objective = model.objective

        with session.graph.as_default(), tf.name_scope(self.name):
            full_var_list = self._gen_var_list(model, var_list)

            # Create optimizer variables before initialization.
            self._minimize_operation = self.optimizer.minimize(
                objective, var_list=full_var_list, **kwargs)

            model.initialize(session=session, force=initialize)
            self._initialize_optimizer(session, full_var_list)
            feed_dict = self._gen_feed_dict(model, feed_dict)
            for _i in range(maxiter):
                session.run(self.minimize_operation, feed_dict=feed_dict)

        if anchor:
            model.anchor(session)

    def _forward_gradients(ys, xs, d_xs):
        """
        Forward-mode pushforward analogous to the pullback defined by tf.gradients.
        With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
        the vector being pushed forward, i.e. this computes (d ys / d xs)^T d_xs.

        :param ys: list of variables being differentiated (tensor)
        :param xs: list of variables to differentiate wrt (tensor)
        :param d_xs: list of gradients to push forward (same shapes as ys)
        :return: the specified moment of the variational distribution
        """
        v = [tf.placeholder(y.dtype) for y in ys]
        g = tf.gradients(ys, xs, grad_ys=v)
        return tf.gradients(g, v, grad_ys=d_xs)

    def _build_natgrad_step_op(objective, q_mu_param, q_sqrt_param, xi_transform=None):
        """
        """
        xi_transforms = XiNat() if xi_transform is None else xi_transform

        q_mu_u, q_sqrt_u = q_mu_param.unconstrained_tensor, q_sqrt_param.unconstrained_tensor
        q_mu, q_sqrt = q_mu_param.constrained_tensor, q_sqrt_param.constrained_tensor

        etas = meanvarsqrt_to_expectation(q_mu, q_sqrt)
        nats = meanvarsqrt_to_natural(q_mu, q_sqrt)

        dL_d_mean, dL_d_varsqrt = tf.gradients(objective, [q_mu, q_sqrt])
        _nats = expectation_to_meanvarsqrt(*etas)
        dL_detas = tf.gradients(_nats, etas, grad_ys=[dL_d_mean, dL_d_varsqrt])

        _xis = xi_transform.naturals_to_xi(*nats)
        nat_dL_xis = forward_gradients(_xis, nats, dL_detas)

        xis = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)

        xis_new = [xis[i] - self.gamma * nat_dL_xis[i] for i in range(2)]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(xis_new)
        mean_new.set_shape(q_mu_param.shape)
        varsqrt_new.set_shape(q_sqrt_param.shape)

        q_mu_assign = tf.assign(q_mu_u, q_mu_param.transform.backward_tensor(mean_new)))
        q_sqrt_assign = tf.assign(q_sqrt_u, q_sqrt_param.transform.backward_tensor(varsqrt_new)))
        return q_mu_assign, q_sqrt_assign


    def _build_natgrad_step_ops(objective, params_and_transforms):
        ops = [_build_natgrad_step_op() in q_mu, q_sqrt, xi_transform in params_and_transforms]
        ops = list(sum(ops, ()))
        return tf.group(ops)
