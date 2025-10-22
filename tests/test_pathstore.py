import math

import pytest

from sssp.pathstore import BlockTree, PathStore

BLOCK_SIZE = 3


@pytest.fixture
def new_tree() -> BlockTree:
    return BlockTree(math.inf, BLOCK_SIZE)


@pytest.fixture
def path_store() -> PathStore:
    return PathStore(math.inf, BLOCK_SIZE)


def test_insert_to_empty_tree(new_tree: BlockTree):
    new_tree.insert(1, 10)

    assert not new_tree.is_empty()
    assert len(new_tree) == 1
    assert new_tree.root.entries[1] == 10


def test_insert_causing_split(new_tree: BlockTree):
    # fill root block
    nodes = [(1, 5), (2, 7), (3, 12)]
    [new_tree.insert(k, v) for k, v in nodes]

    # insert to trigger split
    new_tree.insert(4, 3)

    assert len(new_tree) == 2
    assert len(new_tree.root.entries) == 2
    assert new_tree.root.left is not None
    assert new_tree.root.left.value == 5


def test_search(new_tree: BlockTree):
    nodes = [(1, 4), (2, 7), (3, 9), (4, 5), (5, 7), (6, 12), (7, 8)]
    [new_tree.insert(k, v) for k, v in nodes]

    # resulting tree should have 3 nodes:
    #
    #              B = 7 [(2, 7), (5, 7)]
    #               /                  \
    #  B=5 [(1, 4), (4, 5)]   B=inf [(7, 8), (3, 9), (6, 12)]
    #
    # search finds the block with the smallest bound >= v

    block = new_tree.search(6)

    assert block.size() == 2
    assert block.value == 7
    assert block == new_tree.root
    assert block.left is not None and block.right is not None


def test_batch_prepend(path_store: PathStore):
    # nodes must be a list of (b, a) tuples for lexicographical sorting
    nodes = [(8, 1), (12, 2), (4, 3), (9, 4)]
    path_store.batch_prepend(nodes)

    assert len(path_store.batch_entries) == 4
    assert path_store.batch_entries[0][0] == 4


def test_delete_from_tree(path_store: PathStore):
    nodes = [(1, 5), (2, 7), (3, 12)]
    [path_store.insert(k, v) for k, v in nodes]

    path_store.delete(2, 7)

    assert path_store.block_tree.root.size() == 2
    assert 2 not in path_store.block_tree.root.entries


def test_delete_emptying_block(path_store: PathStore):
    nodes = [(1, 4), (2, 7), (3, 9), (4, 5), (5, 7), (6, 12), (7, 8)]
    [path_store.insert(k, v) for k, v in nodes]

    assert path_store.block_tree.size() == 3

    path_store.delete(2, 7)
    path_store.delete(5, 7)

    assert path_store.block_tree.size() == 2
    assert path_store.block_tree.root.value != 7
