
# pylint: disable=W0212

import unittest
import os
import tensorflow as tf
import gpflow

from gpflow.test_util import GPflowTestCase

class TestConfigParsing(GPflowTestCase):
    def setUp(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        f = os.path.join(directory, 'gpflowrc_test.txt')
        self.conf = gpflow._settings._read_config_file(f)
        self.settings = gpflow._settings._namedtuplify(self.conf._sections)

    def test(self):
        self.assertTrue(all([
            self.settings.first_section.a_bool is False,
            self.settings.first_section.a_float == 1e-3,
            self.settings.first_section.a_string == 'hello',
            self.settings.first_section.a_type is tf.float64,
            self.settings.second_section.a_bool is True,
            self.settings.second_section.another_bool is True,
            self.settings.second_section.yet_another_bool is False]))

    def test_config_not_found(self):
        filename = "./config_not_found.txt"
        with self.assertRaises(RuntimeError):
            gpflow._settings._read_config_file(filename)

    def test_parser(self):
        with self.assertRaises(ValueError):
            gpflow._settings._parse(None)

        with self.assertRaises(ValueError):
            gpflow._settings._parse(12)

        with self.assertRaises(ValueError):
            gpflow._settings._parse([])

        self.assertEqual(gpflow._settings._parse('false'), False)
        self.assertEqual(gpflow._settings._parse('False'), False)
        self.assertEqual(gpflow._settings._parse('true'), True)
        self.assertEqual(gpflow._settings._parse('True'), True)
        self.assertEqual(gpflow._settings._parse('int32'), tf.int32)
        self.assertEqual(gpflow._settings._parse('32'), 32)
        self.assertEqual(gpflow._settings._parse('32.'), 32.)
        self.assertEqual(gpflow._settings._parse('int'), 'int')
        self.assertEqual(gpflow._settings._parse('hello'), 'hello')
        self.assertEqual(gpflow._settings._parse('1E2'), 1e2)
        self.assertEqual(gpflow._settings._parse('1e-9'), 1e-9)


class TestSettingsManager(GPflowTestCase):
    def testRaises(self):
        with self.assertRaises(AttributeError):
            gpflow.settings.undefined_setting_to_raise_error

    def testMutability(self):
        orig = gpflow.settings.verbosity.hmc_verb
        gpflow.settings.verbosity.hmc_verb = False
        self.assertEqual(gpflow.settings.verbosity.hmc_verb, False)
        gpflow.settings.verbosity.hmc_verb = True
        self.assertEqual(gpflow.settings.verbosity.hmc_verb, True)
        gpflow.settings.verbosity.hmc_verb = orig

    def testContextManager(self):
        orig = gpflow.settings.verbosity.hmc_verb
        gpflow.settings.verbosity.hmc_verb = True
        config = gpflow.settings.get_settings()
        config.verbosity.hmc_verb = False
        self.assertEqual(gpflow.settings.verbosity.hmc_verb, True)
        with gpflow.settings.temp_settings(config):
            self.assertEqual(gpflow.settings.verbosity.hmc_verb, False)
        self.assertEqual(gpflow.settings.verbosity.hmc_verb, True)
        gpflow.settings.verbosity.hmc_verb = orig


if __name__ == "__main__":
    unittest.main()
