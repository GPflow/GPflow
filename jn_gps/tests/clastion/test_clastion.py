from jn_gps.clastion import Clastion, derived, put


def test_clastion() -> None:
    ab_count = 0
    abc_count = 0

    class Foo(Clastion):
        a = put(int)
        b = put(int)
        c = put(int)

        @derived()
        def ab(self) -> int:
            nonlocal ab_count
            ab_count += 1
            return self.a + self.b

        @derived()
        def abc(self) -> int:
            nonlocal abc_count
            abc_count += 1
            return self.ab + self.c

    foo = Foo()
    foo = foo(a=3, b=7)
    assert 3 == foo.a
    assert 7 == foo.b
    assert 3 + 7 == foo.ab
    assert 1 == ab_count

    old_foo = foo
    foo = foo(c=4)
    assert 3 + 7 == foo.ab
    assert 1 == ab_count
    assert 3 + 7 + 4 == foo.abc
    assert 1 == abc_count

    foo = foo(a=5)
    assert 5 + 7 == foo.ab
    assert 2 == ab_count
    assert 5 + 7 + 4 == foo.abc
    assert 2 == abc_count

    assert 3 + 7 == old_foo.ab
    assert 2 == ab_count


def test_clastion__defaults() -> None:
    class Foo(Clastion):
        @put()
        def a(self) -> int:
            return 1

        @put()
        def b(self) -> int:
            return 2

        c = put(int)

        @derived()
        def ab(self) -> int:
            return self.a + self.b

        @derived()
        def abc(self) -> int:
            return self.ab + self.c

    foo = Foo(a=5, b=4, c=3)
    assert 5 == foo.a
    assert 4 == foo.b
    assert 5 + 4 == foo.ab
    assert 5 + 4 + 3 == foo.abc

    foo = Foo(b=4, c=3)
    assert 1 == foo.a
    assert 4 == foo.b
    assert 1 + 4 == foo.ab
    assert 1 + 4 + 3 == foo.abc

    foo = Foo(c=3)
    assert 1 == foo.a
    assert 2 == foo.b
    assert 1 + 2 == foo.ab
    assert 1 + 2 + 3 == foo.abc


def test_clastion__nesting() -> None:
    class Bar(Clastion):
        a = put(int)
        b = put(int)

        @derived()
        def s(self) -> int:
            return self.a + self.b

    class Foo(Clastion):
        ab = put(Bar)
        c = put(int)

        @derived()
        def abc(self) -> Bar:
            return Bar(a=self.ab.s, b=self.c)

    foo = Foo()
    foo = foo(ab=Bar(a=3, b=7))
    assert 3 == foo.ab.a
    assert 7 == foo.ab.b
    assert 3 + 7 == foo.ab.s

    old_foo = foo
    foo = foo(c=4)
    assert 3 + 7 == foo.ab.s
    assert 3 + 7 + 4 == foo.abc.s

    foo = foo(ab=foo.ab(a=5))
    assert 5 + 7 == foo.ab.s
    assert 5 + 7 + 4 == foo.abc.s

    assert 3 + 7 == old_foo.ab.s


def test_clastion__repr() -> None:
    class Bar(Clastion):
        a = put(int)
        b = put(int)

        @derived()
        def s(self) -> int:
            return self.a + self.b

    class Foo(Clastion):
        ab = put(Bar)
        c = put(int)
        s = put(str)

        @derived()
        def abc(self) -> Bar:
            return Bar(a=self.ab.s, b=self.c)

    foo = Foo()
    foo = foo(ab=Bar(a=3, b=7))

    assert """Foo(
    ab=Bar(
        a=3,
        b=7,
        s=** Not computed / cached **,
    ),
    abc=** Not computed / cached **,
    c=** Not computed / cached **,
    s=** Not computed / cached **,
)""" == str(
        foo
    )

    foo = foo(c=4, s="a\nb\nc")
    assert """Foo(
    ab=Bar(
        a=3,
        b=7,
        s=** Not computed / cached **,
    ),
    abc=** Not computed / cached **,
    c=4,
    s='a\\nb\\nc',
)""" == str(
        foo
    )

    foo.abc  # pylint: disable=pointless-statement
    assert """Foo(
    ab=Bar(
        a=3,
        b=7,
        s=10,
    ),
    abc=Bar(
        a=10,
        b=4,
        s=** Not computed / cached **,
    ),
    c=4,
    s='a\\nb\\nc',
)""" == str(
        foo
    )
