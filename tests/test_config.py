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


# pylint: disable=W0212

import inspect
import logging
import os

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.test_util import GPflowTestCase


CONFIG_TXT = """
[first_section]
a_bool = false
a_float = 1e-3
a_string = hello
a_type = float64

[second_section]
a_bool = true
another_bool = True
yet_another_bool = False
"""

class TestConfigParsing(GPflowTestCase):
    config_filename = None

    @classmethod
    def setUpClass(cls):
        directory = tf.test.get_temp_dir()
        cls.config_filename = os.path.join(directory, 'gpflowrc_test.txt')
        with open(cls.config_filename, 'w') as fd:
            fd.write(CONFIG_TXT)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.config_filename)
        super().tearDownClass()

    def setUp(self):
        self.conf = gpflow._settings._read_config_file(self.config_filename)
        self.settings = gpflow._settings._namedtuplify(self.conf._sections)

    def test(self):
        self.assertEqual(self.settings.first_section.a_bool, False)
        self.assertEqual(self.settings.first_section.a_float, 1e-3)
        self.assertEqual(self.settings.first_section.a_bool, False)
        self.assertEqual(self.settings.first_section.a_string, 'hello')
        self.assertEqual(self.settings.first_section.a_type, np.float64)
        self.assertEqual(self.settings.second_section.a_bool, True)
        self.assertEqual(self.settings.second_section.another_bool, True)
        self.assertEqual(self.settings.second_section.yet_another_bool, False)

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

    def testDeprecated(self):
        s = gpflow.settings
        self.assertEqual(s.tf_float, s.float_type)
        self.assertEqual(s.np_float, s.float_type)
        self.assertEqual(s.tf_int, s.int_type)
        self.assertEqual(s.np_int, s.int_type)
        with self.assertWarns(DeprecationWarning):
            _ = s.tf_float
        with self.assertWarns(DeprecationWarning):
            _ = s.tf_int
        with self.assertWarns(DeprecationWarning):
            _ = s.np_float
        with self.assertWarns(DeprecationWarning):
            _ = s.np_int

    def testMutability(self):
        orig = gpflow.settings.verbosity.tf_compile_verb
        gpflow.settings.verbosity.tf_compile_verb = False
        self.assertEqual(gpflow.settings.verbosity.tf_compile_verb, False)
        gpflow.settings.verbosity.tf_compile_verb = True
        self.assertEqual(gpflow.settings.verbosity.tf_compile_verb, True)
        gpflow.settings.verbosity.tf_compile_verb = orig

    def testContextManager(self):
        orig = gpflow.settings.verbosity.tf_compile_verb
        gpflow.settings.verbosity.tf_compile_verb = True
        config = gpflow.settings.get_settings()
        config.verbosity.tf_compile_verb = False
        self.assertEqual(gpflow.settings.verbosity.tf_compile_verb, True)
        with gpflow.settings.temp_settings(config):
            self.assertEqual(gpflow.settings.verbosity.tf_compile_verb, False)
        self.assertEqual(gpflow.settings.verbosity.tf_compile_verb, True)
        gpflow.settings.verbosity.tf_compile_verb = orig

def test_logging():
    def level_name(log):
        return logging.getLevelName(log.level)

    warning = 'WARNING'
    assert gpflow.settings.logging_level == warning
    logger = gpflow.settings.logger()
    assert level_name(logger) == warning

    debug = 'DEBUG'
    gpflow.settings.logging.level = debug
    logger = gpflow.settings.logger()
    assert level_name(logger) == debug
    module_name = inspect.getmodule(inspect.currentframe()).__name__
    assert logger.name == module_name


if __name__ == '__main__':
    tf.test.main()
