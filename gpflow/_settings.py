import os
import copy
import collections
import warnings
import logging
import inspect

from collections import OrderedDict
import configparser

import numpy as np
import tensorflow as tf



class _SettingsContextManager(object):
    def __init__(self, manager, tmp_settings):
        self._manager = manager
        self._tmp_settings = tmp_settings

    def __enter__(self):
        self._manager.push(self._tmp_settings)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager.pop()


class _SettingsManager(object):
    def __init__(self, cur_settings):
        self._cur_settings = cur_settings
        self._settings_stack = []

    def __getattr__(self, name):
        try:
            return self._cur_settings[name]
        except KeyError:
            raise AttributeError("Unknown setting.")

    def push(self, extra_settings):
        self._settings_stack.append(self._cur_settings)
        self._cur_settings = extra_settings

    def pop(self):
        rem = self._cur_settings
        self._cur_settings = self._settings_stack.pop()
        return rem

    def temp_settings(self, tmp_settings):
        return _SettingsContextManager(self, tmp_settings)

    def get_settings(self):
        return copy.deepcopy(self._cur_settings)

    @property
    def jitter(self):
        return self.numerics.jitter_level

    @property
    def tf_float(self):
        warnings.warn('tf_float is deprecated and will be removed at GPflow '
                      'version 1.2.0. Use float_type.', DeprecationWarning)
        return self.float_type

    @property
    def tf_int(self):
        warnings.warn('tf_int is deprecated and will be removed at GPflow '
                      'version 1.2.0. Use int_type.', DeprecationWarning)
        return self.int_type

    @property
    def np_float(self):
        warnings.warn('np_float is deprecated and will be removed at GPflow '
                      'version 1.2.0. Use float_type.', DeprecationWarning)
        return self.float_type

    @property
    def np_int(self):
        warnings.warn('np_int is deprecated and will be removed at GPflow '
                      'version 1.2.0. Use int_type.', DeprecationWarning)
        return self.int_type

    @property
    def float_type(self):
        return self.dtypes.float_type

    @property
    def int_type(self):
        return self.dtypes.int_type

    @property
    def logging_level(self):
        return self.logging.level

    def logger(self):
        frame = inspect.currentframe().f_back
        module = inspect.getmodule(frame)
        name = 'gpflow' if module is None else module.__name__
        level = logging.getLevelName(self.logging.level)
        logging.basicConfig()
        log = logging.getLogger(name)
        log.setLevel(level)
        return log


class _MutableNamedTuple(OrderedDict):
    """
    A class that doubles as a mutable named tuple, to allow settings
    to be re-set during
    """
    def __init__(self, *args, **kwargs):
        super(_MutableNamedTuple, self).__init__(*args, **kwargs)
        self._settings_stack = []
        self._initialised = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialised"):
            super(_MutableNamedTuple, self).__setattr__(name, value)
        else:
            super(_MutableNamedTuple, self).__setitem__(name, value)


# a very simple parser
def _parse(string):
    """
    Very simple config values parser.
    """
    if not isinstance(string, str):
        raise ValueError('Config value "{0}" expected to be string.'
                         .format(string))
    if string in ['true', 'True']:
        return True
    elif string in ['false', 'False']:
        return False
    elif string in ['float64', 'float32', 'float16',
                    'int64', 'int32', 'int16']:
        return getattr(np, string)
    else:
        try:
            return int(string)
        except ValueError:
            pass
        try:
            return float(string)
        except ValueError:
            return string


def _namedtuplify(mapping):
    """
    Make the dictionary into a nested series of named tuples.
    This is what allows accessing by attribute: settings.numerics.jitter
    Thank you https://gist.github.com/hangtwenty/5960435
    """
    if isinstance(mapping, collections.Mapping):
        for key, value in list(mapping.items()):
            mapping[key] = _namedtuplify(value)
        try:
            mapping.pop('__name__')
        except KeyError:
            pass
        # return collections.namedtuple('settingsa', dict(**mapping))(**mapping)
        return _MutableNamedTuple(mapping)
    return _parse(mapping)


def _read_config_file(path=None):
    """
    Reads config file.
    First look for config file in the current directory, then in the
    user's home directory, then in the same directory as this file.
    Tries to find config file both with and without preceeding 'dot'
    for hidden files (prefer non-hidden).
    """
    cfg = configparser.ConfigParser()

    if path is None:  # pragma: no cover
        dirs = [os.curdir, os.path.expanduser('~'),
                os.path.dirname(os.path.realpath(__file__))]
        locations = map(os.path.abspath, dirs)
        for loc in locations:
            if cfg.read(os.path.join(loc, 'gpflowrc')):
                break
            if cfg.read(os.path.join(loc, '.gpflowrc')):
                break
    else:
        if not cfg.read(path):
            raise RuntimeError("Config at '{0}' cannot be read".format(path))
    return cfg


__CONFIG = _read_config_file()
__LOADED_SETTINGS = _namedtuplify(__CONFIG._sections)

SETTINGS = _SettingsManager(__LOADED_SETTINGS) # pylint: disable=C0103
