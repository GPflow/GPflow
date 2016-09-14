import unittest
import GPflow
import os
import tensorflow as tf


class TestConfigParsing(unittest.TestCase):
    def setUp(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        f = os.path.join(directory, 'gpflowrc_test.txt')
        self.conf = GPflow._settings.read_config_file(f)
        self.settings = GPflow._settings.namedtuplify(self.conf._sections)

    def test(self):
        self.assertTrue(all([
            self.settings.first_section.a_bool is False,
            self.settings.first_section.a_float == 1e-3,
            self.settings.first_section.a_string == 'hello',
            self.settings.first_section.a_type is tf.float64,
            self.settings.second_section.a_bool is True,
            self.settings.second_section.another_bool is True,
            self.settings.second_section.yet_another_bool is False]))

    def test_parser(self):
        with self.assertRaises(ValueError):
            GPflow._settings.parse(None)

        with self.assertRaises(ValueError):
            GPflow._settings.parse(12)

        with self.assertRaises(ValueError):
            GPflow._settings.parse([])

        self.assertTrue(GPflow._settings.parse('false') is False)
        self.assertTrue(GPflow._settings.parse('False') is False)
        self.assertTrue(GPflow._settings.parse('true') is True)
        self.assertTrue(GPflow._settings.parse('True') is True)
        self.assertTrue(GPflow._settings.parse('int32') is tf.int32)
        self.assertTrue(GPflow._settings.parse('32') is 32)
        self.assertTrue(GPflow._settings.parse('32.') == 32.)
        self.assertTrue(GPflow._settings.parse('int') == 'int')
        self.assertTrue(GPflow._settings.parse('hello') == 'hello')
        self.assertTrue(GPflow._settings.parse('1E2') == 1e2)
        self.assertTrue(GPflow._settings.parse('1e-9') == 1e-9)


class TestConfigContexts(unittest.TestCase):
    def testMutability(self):
        orig = GPflow.settings.verbosity.hmc_verb
        GPflow.settings.verbosity.hmc_verb = False
        self.assertTrue(GPflow.settings.verbosity.hmc_verb is False)
        GPflow.settings.verbosity.hmc_verb = True
        self.assertTrue(GPflow.settings.verbosity.hmc_verb is True)
        GPflow.settings.verbosity.hmc_verb = orig

    def testContextManager(self):
        orig = GPflow.settings.verbosity.hmc_verb
        GPflow.settings.verbosity.hmc_verb = True
        config = GPflow.settings.get_settings()
        config.verbosity.hmc_verb = False
        self.assertTrue(GPflow.settings.verbosity.hmc_verb is True)
        with GPflow.settings.temp_settings(config):
            self.assertTrue(GPflow.settings.verbosity.hmc_verb is False)
        self.assertTrue(GPflow.settings.verbosity.hmc_verb is True)
        GPflow.settings.verbosity.hmc_verb = orig


if __name__ == "__main__":
    unittest.main()
