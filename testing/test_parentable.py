
import unittest

from GPflow import parentable


class TestParentable(unittest.TestCase):
    def test_no_identical_children(self):
        child = parentable.Parentable()

        class MockParent(object):
            def __init__(self):
                self.name_dict = {"child": child, "identical_child": child}

        child._parent = MockParent()
        with self.assertRaises(ValueError):
            child.name
