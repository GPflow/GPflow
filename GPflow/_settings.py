from six.moves import configparser
import copy
import os
import collections
import tensorflow as tf
from collections import OrderedDict


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
            raise AttributeError

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
        c = copy.deepcopy(self._cur_settings)
        c._mutable = True
        return c


class MutableNamedTuple(OrderedDict):
    """
    A class that doubles as a mutable named tuple, to allow settings to be re-set during
    """

    def __init__(self, *args, **kwargs):
        super(MutableNamedTuple, self).__init__(*args, **kwargs)
        self._mutable = True
        self._settings_stack = []

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in ["_settings_stack", "_mutable"]:
            super(MutableNamedTuple, self).__setattr__(name, value)
        elif self._mutable is True:
            super(MutableNamedTuple, self).__setitem__(name, value)
        else:
            raise AttributeError("Instance not mutable.")


# a very simple parser
def parse(string):
    if type(string) is not str:
        raise ValueError
    if string in ['true', 'True']:
        return True
    elif string in ['false', 'False']:
        return False
    elif string in ['float64', 'float32', 'float16', 'int64', 'int32', 'int16']:
        return getattr(tf, string)
    elif any([string.count(s) for s in '.eE']):
        try:
            return float(string)
        except:
            return string
    else:
        try:
            return int(string)
        except:
            return string


# make the dictionary into a nested series of named tuples. This is what allows
# accessing by attribute: settings.numerics.jitter
def namedtuplify(mapping):  # thank you https://gist.github.com/hangtwenty/5960435
    if isinstance(mapping, collections.Mapping):
        for key, value in list(mapping.items()):
            mapping[key] = namedtuplify(value)
        try:
            mapping.pop('__name__')
        except:
            pass
        # return collections.namedtuple('settingsa', dict(**mapping))(**mapping)
        return MutableNamedTuple(mapping)
    return parse(mapping)


def read_config_file(path=None):
    c = configparser.ConfigParser()

    if path is None:  # pragma: no cover
        # first look in the current directory,
        # then in the user's home directory,
        # then in the same directory as this file.
        locations = map(os.path.abspath, [os.curdir,
                                          os.path.expanduser('~'),
                                          os.path.dirname(os.path.realpath(__file__))])
        for loc in locations:
            # try both with and without preceeding 'dot' for hidden files (prefer non-hidden)
            if c.read(os.path.join(loc, 'gpflowrc')):
                break
            if c.read(os.path.join(loc, '.gpflowrc')):
                break
    else:
        assert (c.read(path))
    return c


c = read_config_file()
loaded_settings = namedtuplify(c._sections)
settings = SettingsManager(loaded_settings)
