from typing import Sequence

from jn_gps.clastion import Clastion, Put, derived, put
from jn_gps.clastion.utilities import multi_get, multi_set, root

ELEMENTS = [
    [root.foo.bar[0], root.foo.bar[0]],
    [root.foo.bar[10 ** 8], root.foo.bar[10 ** 8]],
    [root.foo.bar[1], root.foo.bar[1]],
    [root.bar, root.bar],
]


def test_path_element_compare__same() -> None:
    for group in ELEMENTS:
        for i, path_1 in enumerate(group):
            for path_2 in group[i + 1 :]:
                # pylint: disable=superfluous-parens
                assert path_1 == path_2
                assert path_1 <= path_2
                assert not (path_1 < path_2)
                assert path_1 >= path_2
                assert not (path_1 > path_2)
                assert hash(path_1) == hash(path_2)


def test_path_element_compare__different() -> None:
    for i, group_1 in enumerate(ELEMENTS):
        for path_1 in group_1:
            for group_2 in ELEMENTS[i + 1 :]:
                for path_2 in group_2:
                    assert path_1 != path_2
                    assert (path_1 < path_2) or (path_1 > path_2)
                    assert hash(path_1) != hash(path_2)


def test_path_element_compare__sort() -> None:
    all_elements = [p for g in ELEMENTS for p in g]
    all_elements.sort()
    assert all_elements == [
        root.bar,
        root.bar,
        root.foo.bar[0],
        root.foo.bar[0],
        root.foo.bar[1],
        root.foo.bar[1],
        root.foo.bar[10 ** 8],
        root.foo.bar[10 ** 8],
    ]


def test_multi_get_set() -> None:
    class Bar(Clastion):
        a = put(int)
        b = put(int)

        @derived()
        def ab(self) -> int:
            return self.a + self.b

    class Foo(Clastion):
        ab = put(Bar)
        c = put(int)
        l: Put[Sequence[int]] = put()

        @derived()
        def abc(self) -> Bar:
            return Bar(a=self.ab.ab, b=self.c)

    foo = Foo(ab=Bar(a=2, b=3), c=4, l=(5, 6, 7))

    assert {
        root.ab.a: 2,
    } == multi_get(foo, [root.ab.a])
    assert {
        root.ab.b: 3,
    } == multi_get(foo, [root.ab.b])
    assert {
        root.c: 4,
    } == multi_get(foo, [root.c])
    assert {
        root.l[1]: 6,
    } == multi_get(foo, [root.l[1]])
    assert {
        root.ab.a: 2,
        root.ab.b: 3,
        root.c: 4,
    } == multi_get(foo, [root.ab.a, root.ab.b, root.c])

    foo = multi_set(
        foo,
        {
            root.ab.a: 7,
            root.c: 8,
            root.l[1]: 9,
        },
    )

    assert {
        root.ab.a: 7,
        root.ab.b: 3,
        root.c: 8,
        root.l: (5, 9, 7),
        root.abc.ab: 18,
    } == multi_get(foo, [root.ab.a, root.ab.b, root.c, root.l, root.abc.ab])
