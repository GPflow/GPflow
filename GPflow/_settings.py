from six.moves import configparser
import copy
import os
import collections
import tensorflow as tf
from collections import OrderedDict

import logging


class SettingsContextManager(object):
    def __init__(self, manager, tmp_settings):
        self._manager = manager
        self._tmp_settings = tmp_settings

    def __enter__(self):
        self._manager.push(self._tmp_settings)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._manager.pop()


class SettingsManager(object):
    def __init__(self, set):
        self._cur_settings = set
        self._settings_stack = []

    def __getattr__(self, name):
        try:
            return self._cur_settings[name]
        except KeyError:
            raise AttributeError("Unknown setting.")

    def push(self, settings):
        self._settings_stack.append(self._cur_settings)
        self._cur_settings = settings

    def pop(self):
        rem = self._cur_settings
        self._cur_settings = self._settings_stack.pop()
        return rem

    def temp_settings(self, tmp_settings):
        return SettingsContextManager(self, tmp_settings)

    def get_settings(self):
        return copy.deepcopy(self._cur_settings)


class MutableNamedTuple(OrderedDict):
    """
    A class that doubles as a mutable named tuple, to allow settings
    to be re-set during
    """
    def __init__(self, *args, **kwargs):
        super(MutableNamedTuple, self).__init__(*args, **kwargs)
        self._settings_stack = []
        self._initialised = True

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if not hasattr(self, "_initialised"):
            super(MutableNamedTuple, self).__setattr__(name, value)
        else:
            super(MutableNamedTuple, self).__setitem__(name, value)


# a very simple parser
def parse(string):
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
        return getattr(tf, string)
    else:
        try:
            return int(string)
        except ValueError:
            pass
        try:
            return float(string)
        except ValueError:
            return string


def namedtuplify(mapping):
    """
    Make the dictionary into a nested series of named tuples.
    This is what allows accessing by attribute: settings.numerics.jitter
    Thank you https://gist.github.com/hangtwenty/5960435
    """
    if isinstance(mapping, collections.Mapping):
        for key, value in list(mapping.items()):
            mapping[key] = namedtuplify(value)
        try:
            mapping.pop('__name__')
        except KeyError:
            pass
        # return collections.namedtuple('settingsa', dict(**mapping))(**mapping)
        return MutableNamedTuple(mapping)
    return parse(mapping)


def read_config_file(path=None):
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
            raise RuntimeError("Config at '{0}'cannot be read".format(path))
    return cfg


config = read_config_file()
loaded_settings = namedtuplify(config._sections)
settings = SettingsManager(loaded_settings)
