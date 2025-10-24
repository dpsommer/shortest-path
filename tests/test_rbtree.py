import pytest

from sssp import rbtree as rb


@pytest.fixture
def tree():
    yield rb.RedBlackTree()


@pytest.mark.parametrize(
        "values,rmv,expected", [
            ([10, 12, 8], 10, 12),
            ([10, 12], 10, 12),
            ([10], 10, None),
            ([10, 12, 8], 8, 10),
            ([10, 12, 8, 6], 8, 10),
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 6, 4)
        ],
        ids=[
            "two_children",
            "one_child",
            "no_children_root",
            "no_children_red",
            "simple_rebalance",
            "complex_rebalance"
        ]
)
def test_remove(tree: rb.RedBlackTree, values, rmv, expected):
    for val in values:
        parent = tree.root
        direction = rb.Direction.LEFT

        if parent:
            direction = rb.Direction(int(val > parent.value))
            child = parent.get_child(direction)
            while child is not None:
                parent = child
                direction = rb.Direction(int(val > parent.value))
                child = parent.get_child(direction)

        tree.insert(rb.Node(val), parent, direction)

    to_remove = tree.search(rmv)
    tree.remove(to_remove)

    print(tree.pprint(tree.root))

    if expected is None:
        assert tree.root is None
    else:
        assert tree.root.value == expected


def test_rotation():
    pass


def test_insert():
    pass
