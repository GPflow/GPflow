import warnings

import multipledispatch

import gpflow


class A1:
    pass


class A2(A1):
    pass


class B1:
    pass


class B2(B1):
    pass


def test_our_multipledispatch():
    test_fn = gpflow.utilities.Dispatcher('test_fn')

    @test_fn.register(A1, B1)
    def test_a1_b1(x, y):
        return 'a1-b1'

    @test_fn.register(A2, B1)
    def test_a2_b1(x, y):
        return 'a2-b1'

    @test_fn.register(A1, B2)
    def test_a1_b2(x, y):
        return 'a1-b2'

    assert test_fn(A1(), B1()) == 'a1-b1'
    assert test_fn(A2(), B1()) == 'a2-b1'
    assert test_fn(A1(), B2()) == 'a1-b2'

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        assert test_fn(A2(), B2()) == 'a1-b2'

        assert len(w) == 1
        assert issubclass(w[0].category, multipledispatch.conflict.AmbiguityWarning)

    @test_fn.register(A2, B2)
    def test_a2_b2(x, y):
        return 'a2-b2'

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        assert test_fn(A2(), B2()) == 'a2-b2'

        assert len(w) == 0
