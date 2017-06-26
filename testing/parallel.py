import unittest


class ParallelTestCase(unittest.TestCase):
    _multiprocess_can_split_ = True