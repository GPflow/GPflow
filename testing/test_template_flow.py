import uuid

import tensorflow as tf

import gpflow
from gpflow.test_util import GPflowTestCase

class DumbModel(gpflow.models.Model):
    def __init__(self):
        gpflow.models.Model.__init__(self)
        self.var_name = str(uuid.uuid4()).replace('-', '_')  # just want a unique name!
        self.a = gpflow.Param(3.)

    @gpflow.autoflow((gpflow.settings.tf_float, []))
    def return_cost1(self, x):
        return self._cost(x)

    @gpflow.autoflow((gpflow.settings.tf_float, []))
    def return_cost2(self, x):
        return self._cost(x)

    @gpflow.templateflow("cost_in_dumb")
    def _cost(self, x):
        var = tf.get_variable(self.var_name, initializer=tf.constant([23.],
                              dtype=gpflow.settings.tf_float),
                              trainable=True, dtype=gpflow.settings.tf_float)
        return x * var + var**2

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return -tf.square(self.a)


class TestTemplateFlow(GPflowTestCase):
    def test_template_reused(self):
        """
        Tests that the template is reused by measuring counts of the variable that we hope that
        Tensorflow resuses. This should be zero even after compile as we have lazy template
        evaluation. However, once it is created then more should not be created even on subsequent
        calls to this function.
        """
        model = DumbModel()
        model.compile()

        def collect_var(name):
            return [v for v in tf.global_variables() if name in v.name]

        vars = collect_var(model.var_name)
        self.assertTrue(len(vars) == 0)  # check has not created it yet

        model.return_cost1(10)
        vars = collect_var(model.var_name)
        self.assertTrue(len(vars) == 1)  # check one of these variables exist

        model.return_cost2(10)
        model.return_cost1(10)
        vars = collect_var(model.var_name)
        self.assertTrue(len(vars) == 1)  # check still only one of these variables exist

    def test_change_echoes_across(self):
        """
        checks that the same variable is used by the same method
        """
        model = DumbModel()
        model.compile()

        def check_model_answers_same(model, x, true_val):
            return model.return_cost2(x) == model.return_cost1(x) == true_val

        self.assertTrue(check_model_answers_same(model, 10, 10*23+23**2))

        # We're gonna get a reference to this variable via its name!
        # which as we've used a uuid _should_ be unique.
        variable = [v for v in tf.global_variables() if model.var_name in v.name][0]
        var_value_new = -2.65
        variable.load([-2.65], model.session)

        self.assertTrue(check_model_answers_same(model, 10, 10*var_value_new+var_value_new**2))


