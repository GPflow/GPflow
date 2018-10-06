# Copyright 2017 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest import TestCase
import mock
from typing import Optional, Dict, Callable
from collections import namedtuple
import tempfile
import pathlib

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import gpflow
import gpflow.actions
import gpflow.training.monitor as mon
from gpflow.test_util import session_context


class _DummyMonitorTask(mon.MonitorTask):

    def __init__(self):
        super().__init__(True)
        self.call_count = 0

    def run(self, context: mon.MonitorContext, *args, **kwargs):
        self.call_count += 1


class DummyLinearModel(gpflow.models.Model):

    def __init__(self, x: np.ndarray, y: np.ndarray,
                w: Optional[np.ndarray]=None, b: Optional[float]=0.0,
                var: Optional[float]=0.0) -> None:

        super().__init__()
        # X is a data matrix; each row represents one instance
        self.X = gpflow.params.DataHolder(x)
        # Y is a data matrix, rows correspond to the rows in X
        self.Y = gpflow.params.DataHolder(y)
        if w is None:
            w = np.ones(x.shape[1:], dtype=np.float)
        self.w = gpflow.params.Parameter(w)
        self.b = gpflow.params.Parameter(b)
        self.var = gpflow.params.Parameter(var)

    @gpflow.decors.params_as_tensors
    def _build_likelihood(self):
        w = tf.expand_dims(self.w, 0)
        f = tf.matmul(self.X, w, transpose_b=True) + self.b
        return tf.reduce_sum(gpflow.logdensities.gaussian(self.Y, f, self.var))


class TestMonitor(TestCase):

    @mock.patch('gpflow.training.monitor.get_hr_time')
    def test_on_iteration_timing(self, mock_timer):
        """
        Tests how the Monitor keeps track of the total running time and total optimisation time.
        """
        mock_timer.side_effect = [1.0, 3.5, 4.0, 6.0, 7.0]
        context = mon.MonitorContext()
        monitor = mon.Monitor([], context=context)
        # In each call to the _on_iteration the timer is called twice - at the beginning and at
        # the end of the call.
        monitor._on_iteration()
        self.assertEqual(monitor._context.total_time, 2.5)
        self.assertEqual(monitor._context.optimisation_time, 2.5)
        monitor._on_iteration()
        self.assertEqual(monitor._context.total_time, 5.0)
        self.assertEqual(monitor._context.optimisation_time, 4.5)


class TestMonitorTask(TestCase):

    @mock.patch('gpflow.training.monitor.get_hr_time')
    def test_call_timing(self, mock_timer):
        """
        Test how a monitoring task keeps track of the last execution time and accumulated execution
        time.
        """
        mock_timer.side_effect = [1.0, 3.5, 4.0, 6.0]
        monitor_task = _DummyMonitorTask()
        monitor_context = mon.MonitorContext()
        monitor_task(monitor_context)
        self.assertEqual(monitor_task.total_time, 2.5)
        self.assertEqual(monitor_task.last_call_time, 2.5)
        monitor_task(monitor_context)
        self.assertEqual(monitor_task.total_time, 4.5)
        self.assertEqual(monitor_task.last_call_time, 2.0)

    def test_call_condition(self):
        """
        Tests that the execution of a task is controlled by the task condition.
        """
        monitor_task = _DummyMonitorTask().with_condition(
            lambda context: context.iteration_no % 2 == 0)
        monitor_context = mon.MonitorContext()
        for monitor_context.iteration_no in range(5):
            monitor_task(monitor_context)
        self.assertEqual(monitor_task.call_count, 3)

    def test_exit_condition(self):
        """
        Tests that the execution of a task after the optimisation is finished is controlled by
        the exit condition.
        """
        monitor_task1 = _DummyMonitorTask().with_exit_condition(False)
        monitor_task2 = _DummyMonitorTask().with_exit_condition(True)
        monitor_context = mon.MonitorContext()
        monitor_context.optimisation_finished = True
        monitor_task1(monitor_context)
        monitor_task2(monitor_context)
        self.assertEqual(monitor_task1.call_count, 0)
        self.assertEqual(monitor_task2.call_count, 1)


class TestGenericCondition(TestCase):

    def test_condition(self):
        """
        Tests generic condition on arbitrary sequence
        """
        sequence = iter([2, 5, 6, 9])
        monitor_context = mon.MonitorContext()
        condition = mon.GenericCondition(lambda context: context.iteration_no, sequence)
        # Input data in the format
        # (expected condition._next, context.iteration_no, condition value)
        steps = [(2, 1, False), (2, 3, True), (5, 4, False), (5, 7, True), (9, 8, False)]
        for expected_next, iter_no, expected_result in steps:
            self.assertEqual(condition._next, expected_next)
            monitor_context.iteration_no = iter_no
            self.assertEqual(condition(monitor_context), expected_result)


class TestPeriodicIterationCondition(TestCase):

    def test_condition(self):
        """
        Tests periodic condition based on the iteration number
        """
        monitor_context = mon.MonitorContext()
        condition = mon.PeriodicIterationCondition(5)
        count = 0
        for monitor_context.iteration_no in range(37):
            if condition(monitor_context):
                count += 1
        self.assertEqual(count, 7)


class TestGrowingIntervalCondition(TestCase):

    def test_sequence(self):
        """
        Tests growing step sequence with no initial value
        """
        seq_iterator = mon.GrowingIntervalCondition._growing_step_sequence(
            interval_growth=2.0, max_interval=10.0, init_interval=3.0)
        expected_sequence = [3.0, 9.0, 19.0, 29.0]
        self.assertListEqual(expected_sequence, [next(seq_iterator) for _ in range(4)])

    def test_sequence_with_init_value(self):
        """
        Tests growing step sequence with initial value
        """
        seq_iterator = mon.GrowingIntervalCondition._growing_step_sequence(
            interval_growth=2.0, max_interval=10.0, init_interval=3.0, start_level=1.0)
        expected_sequence = [1.0, 7.0, 17.0, 27.0]
        self.assertListEqual(expected_sequence, [next(seq_iterator) for _ in range(4)])


class TestPrintTimingsTask(TestCase):

    def test_print_timings(self):
        """
        Tests rate calculation for the PrintTimingsTask (doesn't test the actual printing)
        """
        with session_context(tf.Graph()):
            monitor_task = mon.PrintTimingsTask()
            monitor_task._print_timings = mock.MagicMock()
            monitor_context = mon.MonitorContext()
            monitor_context.session = tf.Session()
            monitor_context.global_step_tensor = mon.create_global_step(monitor_context.session)
            monitor_context.init_global_step = 100

            # First call
            monitor_context.iteration_no = 10
            monitor_context.total_time = 20.0
            monitor_context.optimisation_time = 16.0
            monitor_context.session.run(monitor_context.global_step_tensor.assign(150))
            monitor_task(monitor_context)
            args = monitor_task._print_timings.call_args_list[0][0]
            self.assertTupleEqual(args, (10, 150, 0.5, 0.5, 3.125, 3.125))

            # Second call
            monitor_context.iteration_no = 24
            monitor_context.total_time = 30.0
            monitor_context.optimisation_time = 24.0
            monitor_context.session.run(monitor_context.global_step_tensor.assign(196))
            monitor_task(monitor_context)
            args = monitor_task._print_timings.call_args_list[1][0]
            self.assertTupleEqual(args, (24, 196, 0.8, 1.4, 4.0, 5.75))


class TestCallbackTask(TestCase):

    def test_callback(self):

        callback = mock.MagicMock()
        monitor_task = mon.CallbackTask(callback)
        monitor_task(mon.MonitorContext())
        self.assertEqual(callback.call_count, 1)


class TestSleepTask(TestCase):

    def test_sleep_lower_bound(self):
        """
        Test that the sleep task breaks the execution for at least the required period of time
        (up to certain precision).
        """
        monitor_task = mon.SleepTask(0.2)
        start_time = mon.get_hr_time()
        monitor_task(mon.MonitorContext())
        elapsed = mon.get_hr_time() - start_time
        self.assertGreater(elapsed, 0.1)


class TestCheckpointTask(TestCase):

    def test_checkpoint_without_global_step(self):
        """
        Tests that saving and restoring a session works. Do not use the global_step which means
        TF won't create multiple checkpoints.
        """
        self._test_chechpoint_roundtrip(False)

    def test_checkpoint_with_global_step(self):
        """
        Tests that saving and restoring a session works. Use the global_step which means TF will
        create multiple checkpoints. The latest checkpoint should be restored.
        """
        self._test_chechpoint_roundtrip(True)

    def _test_chechpoint_roundtrip(self, use_global_step: bool, num_checkpoints: Optional[int]=5):
        """
        Performs saving/restoring roundtrip, either with or without using `global_step`.
        Note that if `global_step` is used the save will create one checkpoint for each value
        of the global step.
        """

        with tempfile.TemporaryDirectory() as tmp_event_dir:

            # Create a variable and do several checkpoints
            with session_context(tf.Graph()) as session:
                dummy_var = self._create_dummy_variable(session)
                monitor_context = mon.MonitorContext()
                monitor_context.session = session
                if use_global_step:
                    monitor_context.global_step_tensor = mon.create_global_step(session)
                monitor_task = mon.CheckpointTask(tmp_event_dir)

                for i in range(num_checkpoints):
                    session.run(dummy_var.assign(i))
                    if use_global_step:
                        session.run(monitor_context.global_step_tensor.assign(10 * i))
                    monitor_task(monitor_context)

            # Restore the session and read the variables.
            # Verify if the latest checkpoint was restored.
            with session_context(tf.Graph()) as session:
                dummy_var = self._create_dummy_variable(session)
                global_step_tensor = mon.create_global_step(session) if use_global_step else None
                mon.restore_session(session, tmp_event_dir)
                self.assertEqual(session.run(dummy_var), num_checkpoints - 1)
                if use_global_step:
                    self.assertEqual(session.run(global_step_tensor), 10 * (num_checkpoints - 1))

    @staticmethod
    def _create_dummy_variable(session: tf.Session):

        dummy_var = tf.Variable(0, name='dummy_var', dtype=tf.int32)
        session.run(tf.variables_initializer([dummy_var]))
        return dummy_var


class TestLogdirWriter(TestCase):

    def test_create_no_error(self):
        """
        Tests that it is possible to create multiple LogdirWriters so long as they write to
        different directories or have different file suffixes.
        """
        with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
            writer1 = mon.LogdirWriter(tmp_dir1)
            writer2 = mon.LogdirWriter(tmp_dir2)
            writer3 = mon.LogdirWriter(tmp_dir2, filename_suffix='suffix')
            writer1.close()
            writer2.close()
            writer3.close()

    def test_reuse_location_no_error(self):
        """
        Tests that it is possible to reuse the location if the original writer is closed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = mon.LogdirWriter(tmp_dir)
            writer.close()
            writer = mon.LogdirWriter(tmp_dir)
            writer.close()

    def test_reopen_writer_no_error(self):
        """
        Tests that it is possible to close and then reopen a writer if its location has not
        been taken by another writer.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = mon.LogdirWriter(tmp_dir)
            writer.close()
            writer.reopen()
            writer.close()

    def test_create_error(self):
        """
        Tests that an attempt to create two writers with the same location causes an error.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            _ = mon.LogdirWriter(tmp_dir, filename_suffix='suffix')
            with self.assertRaises(RuntimeError):
                _ = mon.LogdirWriter(tmp_dir, filename_suffix='suffix')

    def test_reopen_error(self):
        """
        Tests that an attempt to reopen a writer causes an error if the writer's location has
        been taken by another writer.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = mon.LogdirWriter(tmp_dir, filename_suffix='suffix')
            writer.close()
            _ = mon.LogdirWriter(tmp_dir, filename_suffix='suffix')
            with self.assertRaises(RuntimeError):
                writer.reopen()


class TestModelToTensorBoardTask(TestCase):

    def test_std_tensorboard_only_scalars(self):
        """
        Tests the standard tensorboard task with scalar parameters only
        """

        with session_context(tf.Graph()):
            model = create_linear_model()

            def task_factory(writer: mon.LogdirWriter):
                return mon.ModelToTensorBoardTask(writer, model, only_scalars=True)

            summary = run_tensorboard_task(task_factory)
            self.assertAlmostEqual(summary['DummyLinearModel/b'].simple_value, float(model.b.value))
            self.assertAlmostEqual(summary['DummyLinearModel/var'].simple_value,
                                   float(model.var.value))
            self.assertAlmostEqual(summary['optimisation/likelihood'].simple_value,
                                   model.compute_log_likelihood(), places=5)
            self.assertNotIn('DummyLinearModel/w', summary.keys())

    def test_std_tensorboard_all_parameters(self):
        """
        Tests the standard tensorboard task with all parameters and extra summaries
        """
        with session_context(tf.Graph()):
            model = create_linear_model()

            def task_factory(writer: mon.LogdirWriter):
                # create 2 extra summaries
                dummy_vars = [tf.Variable(5.0), tf.Variable(6.0)]
                dummy_vars_init = tf.variables_initializer(dummy_vars)
                model.enquire_session().run(dummy_vars_init)
                add_summaries = [tf.summary.scalar('dummy' + str(i), dummy_var)
                                 for i, dummy_var in enumerate(dummy_vars)]
                return mon.ModelToTensorBoardTask(writer, model, only_scalars=False,
                                                  additional_summaries=add_summaries)

            summary = run_tensorboard_task(task_factory)
            self.assertAlmostEqual(summary['dummy0'].simple_value, 5.0)
            self.assertAlmostEqual(summary['dummy1'].simple_value, 6.0)
            self.assertIn('DummyLinearModel/w', summary.keys())


class TestLmlToTensorBoardTask(TestCase):

    def test_lml_tensorboard(self):
        """
        Tests the LML tensorboard task
        """
        with session_context(tf.Graph()):
            # Create a number of models with the same set of parameters and equal number of
            # data points except one. The data from these model will mimic mini-batches.
            mini_batches = 10
            complete_size = 12
            incomplete_size = 7
            mini_batch_sizes = [complete_size if i < mini_batches - 1 else incomplete_size
                                for i in range(mini_batches)]
            mini_batch_data = [create_leaner_model_data(size) for size in mini_batch_sizes]
            mini_models = [DummyLinearModel(d.x, d.y, d.w, d.b, d.var)
                           for d in mini_batch_data]
            # Calculate average log likelihood across all models
            avg_lml = sum(mdl.compute_log_likelihood() * size
                          for mdl, size in zip(mini_models, mini_batch_sizes))
            avg_lml /= sum(mini_batch_sizes)

            # Join together the datasets from all mini-batch models
            xs = np.concatenate(tuple(d.x for d in mini_batch_data))
            ys = np.concatenate(tuple(d.y for d in mini_batch_data))

            # Create model with the same parameters and joint datasets
            d = mini_batch_data[0]
            model = DummyLinearModel(xs, ys, d.w, d.b, d.var)

            def task_factory(writer: mon.LogdirWriter):
                return mon.LmlToTensorBoardTask(writer, model, minibatch_size=complete_size,
                                                display_progress=False)

            # Run LML task, extract the LML value and compare with the one computed over models with
            # small data sets
            summary = run_tensorboard_task(task_factory)
            self.assertAlmostEqual(summary['DummyLinearModel/full_lml'].simple_value, avg_lml,
                                   places=5)


class TestScalarFuncToTensorBoardTask(TestCase):

    def test_scalar_tensorboard(self):
        """
        Tests Scalar function tensorboard task.
        """

        user_func_name = 'test_scalar_function'
        user_func_value = 5.55

        def user_func(*args, **kwargs):
            return user_func_value

        def task_factory(writer: mon.LogdirWriter):
            return mon.ScalarFuncToTensorBoardTask(writer, user_func, user_func_name)

        summary = run_tensorboard_task(task_factory)
        self.assertAlmostEqual(summary[user_func_name].simple_value, user_func_value, places=5)


class TestVectorFuncToTensorBoardTask(TestCase):

    def test_vector_tensorboard(self):
        """
        Tests Vector function tensorboard task.
        """

        user_func_name = 'test_vector_function'
        user_func_values = [3.3, 4.4, 5.5]

        def user_func(*args, **kwargs):
            return user_func_values

        def task_factory(writer: mon.LogdirWriter):
            return mon.VectorFuncToTensorBoardTask(writer, user_func, user_func_name,
                                                   len(user_func_values))

        summary = run_tensorboard_task(task_factory)
        for i, func_value in enumerate(user_func_values):
            self.assertAlmostEqual(summary[user_func_name + '_' + str(i)].simple_value,
                                   func_value, places=5)


class TestHistogramToTensorBoardTask(TestCase):

    def test_histogram_tensorboard(self):
        """
        Tests Histogram function tensorboard task. Just checks that the histogram summary object
        is created.
        """

        user_func_name = 'test_histogram_function'
        user_func_values = [[1.1, 1.2], [2.1, 2.2], [3.1, 3.3]]

        def user_func(*args, **kwargs):
            return user_func_values

        def task_factory(writer: mon.LogdirWriter):
            return mon.HistogramToTensorBoardTask(writer, user_func, user_func_name,
                                                np.array(user_func_values).shape)

        summary = run_tensorboard_task(task_factory)
        self.assertIsNotNone(summary[user_func_name].histo)


class TestImageToTensorBoardTask(TestCase):

    def test_image_tensorboard(self):
        """
        Tests Matplotlib image tensorboard task. Just checks that the image summary object
        is created
        """

        plot_func_name = 'test_plot_function'

        def plot_func(*args, **kwargs):
            x = np.linspace(0, 2, 100)
            plt.plot(x, x, label='linear')
            plt.plot(x, x ** 2, label='quadratic')
            plt.plot(x, x ** 3, label='cubic')
            return plt.figure()

        def task_factory(writer: mon.LogdirWriter):
            return mon.ImageToTensorBoardTask(writer, plot_func, plot_func_name)

        summary = run_tensorboard_task(task_factory)
        self.assertIsNotNone(summary[plot_func_name + '/image/0'].image)


class TestMonitorIntegration(TestCase):

    def test_with_tensorflow_optimiser(self):
        """
        Tests the monitor with a tensorflow optimiser
        """

        def optimise(model, step_callback, global_step_tensor) -> None:
            """
            Optimisation function that creates and calls the tensorflow AdamOptimizer optimiser.
            """
            optimiser = gpflow.train.AdamOptimizer(0.01)
            optimiser.minimize(model, maxiter=10, step_callback=step_callback,
                               global_step=global_step_tensor)

        with session_context(tf.Graph()):
            self._optimise_model(create_linear_model(), optimise, True)

    @mock.patch('gpflow.training.monitor.update_optimiser')
    def test_with_scipy_optimiser(self, update_optimiser):
        """
        Tests the monitor with the Scipy optimiser
        """

        optimiser = gpflow.train.ScipyOptimizer()

        def optimise(model, step_callback, _) -> None:
            """
            Optimisation function that creates and calls ScipyOptimizer optimiser.
            """
            nonlocal optimiser
            optimiser.minimize(model, maxiter=10, step_callback=step_callback)

        with session_context(tf.Graph()):
            self._optimise_model(create_linear_model(), optimise, optimiser=optimiser)

        self.assertGreater(update_optimiser.call_count, 0)

    def test_with_natgrad_optimiser(self):
        """
        Test the monitor with the Natural Gradient optimiser.
        """

        def optimise(model, step_callback, _) -> None:
            """
            Optimisation function that creates and calls NatGradPtimizer optimiser.
            """
            var_list = [(model.q_mu, model.q_sqrt)]
            # we don't want adam optimizing these
            model.q_mu.set_trainable(False)
            model.q_sqrt.set_trainable(False)

            optimiser = gpflow.train.NatGradOptimizer(1.0)
            optimiser.minimize(model, maxiter=10, var_list=var_list, step_callback=step_callback)

        with session_context(tf.Graph()):
            # NatGrad optimiser works only with variational parameters. So we can't use the
            # dummy linear model here.
            model_data = create_leaner_model_data(20)
            z = np.linspace(0, 1, 5)[:, None]
            model = gpflow.models.SVGP(model_data.x, model_data.y, gpflow.kernels.RBF(1),
                                       gpflow.likelihoods.Gaussian(), Z=z)
            self._optimise_model(model, optimise)

    def test_update_scipy_optimiser(self):
        """
        Checks that the `update_optimiser` function sets the ScipyOptimizer state to the model
        parameters. Also checks that it sets the `optimiser_updated` flag to True.
        """

        with session_context(tf.Graph()):
            model = create_linear_model()
            optimiser = gpflow.train.ScipyOptimizer()
            context = mon.MonitorContext()
            context.session = model.enquire_session()
            context.optimiser = optimiser
            w, b, var = model.w.value, model.b.value, model.var.value
            call_count = 0

            def step_callback(*args, **kwargs):
                nonlocal model, optimiser, context, w, b, var, call_count
                context.optimiser_updated = False
                mon.update_optimiser(context, *args, **kwargs)
                w_new, b_new, var_new = model.enquire_session().run([model.w.unconstrained_tensor,
                                                                     model.b.unconstrained_tensor,
                                                                     model.var.unconstrained_tensor])
                self.assertTrue(np.alltrue(np.not_equal(w, w_new)))
                self.assertTrue(np.alltrue(np.not_equal(b, b_new)))
                self.assertTrue(np.alltrue(np.not_equal(var, var_new)))
                self.assertTrue(context.optimiser_updated)
                call_count += 1
                w, b, var = w_new, b_new, var_new

            optimiser.minimize(model, maxiter=10, step_callback=step_callback)
            self.assertGreater(call_count, 0)

    def _optimise_model(self, model: gpflow.models.Model,
                        optimise_func: Callable[[gpflow.models.Model, Callable, tf.Variable], None],
                        use_global_step: Optional[bool]=False, optimiser=None) -> None:
        """
        Runs optimisation test with given model and optimisation function.
        :param model: Model derived from `gpflow.models.Model`
        :param optimise_func: Function that performs the optimisation. The function should take
        the model, step callback and the `global_step` tensor as the arguments
        :param use_global_step: flag indicating the the `global_step` variable should be used
        """

        session = model.enquire_session()
        global_step_tensor = mon.create_global_step(session) if use_global_step else None

        monitor_task = _DummyMonitorTask()

        lml_before = model.compute_log_likelihood()

        # Run optimisation
        with mon.Monitor([monitor_task], session, global_step_tensor, optimiser=optimiser) \
                as monitor:
            optimise_func(model, monitor, global_step_tensor)

        lml_after = model.compute_log_likelihood()

        if use_global_step:
            # Check that the 'global_step' has the actual number of iterations
            global_step = session.run(global_step_tensor)
            self.assertEqual(global_step, monitor_task.call_count)
        else:
            # Just check that there were some iterations
            self.assertGreater(monitor_task.call_count, 0)

        # Check that the optimiser has done something
        # self.assertGreater(lml_after, lml_before)


LinearModelSetup = namedtuple('LinearModelSetup', ['w', 'b', 'var', 'x', 'y'])


def create_linear_model(data_points: Optional[int]=10) -> gpflow.models.Model:
    """
    Creates an instance of the dummy linear model
    """
    d = create_leaner_model_data(data_points)
    return DummyLinearModel(d.x, d.y, d.w, d.b, d.var)


def create_leaner_model_data(data_points) -> LinearModelSetup:
    """
    Creates data for the dummy linear model with required number of data points
    """
    w = np.array([0.7, 1.3])
    b = 2.0
    var = 0.2
    x = np.random.rand(data_points, 2)
    y = np.expand_dims(np.random.normal(np.matmul(x, np.transpose(w)) + b, np.sqrt(var)), -1)

    return LinearModelSetup(w=w, b=b, var=var, x=x, y=y)


def run_tensorboard_task(task_factory: Callable[[mon.LogdirWriter],
                                                mon.BaseTensorBoardTask]) -> Dict:
    """
    Runs a tensorboard monitoring task, reads summary from the created event file and returns
    decoded proto values in a dictionary
    :param task_factory: task factory that takes the event directory as an argument.
    """

    summary = {}

    with tempfile.TemporaryDirectory() as tmp_event_dir:

        writer = mon.LogdirWriter(tmp_event_dir)
        try:
            monitor_task = task_factory(writer)

            session = monitor_task.model.enquire_session()\
                if monitor_task.model is not None else tf.Session()
            global_step_tensor = mon.create_global_step(session)

            monitor_task.with_flush_immediately(True)

            monitor_context = mon.MonitorContext()
            monitor_context.session = session
            monitor_context.global_step_tensor = global_step_tensor

            monitor_task(monitor_context)

            # There should be one event file in the temporary directory
            event_file = str(next(pathlib.Path(tmp_event_dir).iterdir().__iter__()))

            for e in tf.train.summary_iterator(event_file):
                for v in e.summary.value:
                    summary[v.tag] = v
        finally:
            writer.close()

    return summary
