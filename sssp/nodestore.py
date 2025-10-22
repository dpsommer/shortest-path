import enum
import heapq
import itertools
import math
from typing import Optional


class Direction(enum.IntEnum):
    LEFT = 0
    RIGHT = 1


class Colour(enum.Enum):
    BLACK = 0
    RED = 1


def median_split(batch: list, size: int):
    """Bisects list repeatedly until all segments have `size` elements or fewer

    Args:
        batch (list): list of elements to split
        size (int): max size of sublists

    Returns:
        list: list of bisected sublists
    """
    if len(batch) <= size:
        return [batch]

    mid = math.ceil(len(batch) / 2)
    return median_split(batch[mid:], size) + median_split(batch[:mid], size)


class Node:

    def __init__(self, upper_bound: float):
        self.parent: Optional[Node] = None
        self.right: Optional[Node] = None
        self.left: Optional[Node] = None
        self.colour = Colour.BLACK
        self.value = upper_bound
        self.entries = {}

    def get_child(self, direction: Direction):
        if direction == Direction.LEFT:
            return self.left
        return self.right

    def set_child(self, direction: Direction, node: Optional["Node"]):
        if direction == Direction.LEFT:
            self.left = node
        else:
            self.right = node

    def get_direction(self) -> Optional[Direction]:
        if self.parent is None:
            return None
        return Direction.LEFT if self == self.parent.left else Direction.RIGHT


class BlockTree:

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self._upper_bound = upper_bound
        self.root = Node(upper_bound)

    def is_empty(self):
        return not self.root.entries

    def _insert(self, node: Node, parent: Node, direction: Direction):
        node.colour = Colour.RED
        node.parent = parent

        # the simplest case is that the parent is empty and we add at the root.
        # we should never hit this case since insert() already accounts for it,
        # but it doesn't hurt to have this as a backstop/for future use
        if parent is None:
            self.root = node
            return

        parent.set_child(direction, node)

        # rebalance the tree iteratively
        while parent is not None:
            # the next simplest case - if we already adhere to the tree
            # invariant, no need to do anything else
            if parent.colour == Colour.BLACK:
                return

            # if the parent is the root, set its colour to black and we're done
            grandparent = parent.parent
            if grandparent is None:
                parent.colour = Colour.BLACK
                return

            direction = parent.get_direction()
            uncle = grandparent.get_child(Direction(1 - direction))

            # if the uncle is null, we can rotate the subtree such that the
            # parent and grandparent swap levels, reducing the tree size.
            # if the parent and uncle are different colours, we correct the
            # tree invariant by rotating the parent up and the uncle down
            if uncle is None or uncle.colour == Colour.BLACK:
                # if the inserted value is between the parent and grandparent,
                # we do an extra rotation to swap them
                if node == parent.get_child(Direction(1 - direction)):
                    self._rotate_subtree(parent, direction)
                    node = parent
                    parent = grandparent.get_child(direction)
                # rotate the subtree starting at the inserted node's
                # grandparent to balance the tree
                self._rotate_subtree(grandparent, Direction(1 - direction))
                parent.colour = Colour.BLACK
                grandparent.colour = Colour.RED
                return

            # adjust the colours of the subtree starting at the inserted node's
            # grandparent, then iterate two steps up the tree towards the root
            parent.colour = Colour.BLACK
            uncle.colour = Colour.BLACK
            grandparent.colour = Colour.RED
            node = grandparent
            parent = node.parent
        return

    # Split When a block in D1 exceeds M elements, we perform a split. First, we
    #   identify the median element within the block in O(M) time, partitioning the
    #   elements into two new blocks each with at most ⌈M/2⌉ elements — elements
    #   smaller than the median are placed in the first block, while the rest are
    #   placed in the second. This split ensures that each new block retains about
    #   ⌈M/2⌉ elements while preserving inter-block ordering, so the number of
    #   blocks in D1 is bounded by O(N/M). (Every block in D1 contains Θ(M)
    #   elements, including the elements already deleted.) After the split, we make
    #   the appropriate changes in the binary search tree of upper bounds in
    #   O(max{1, log(N/M)}) time.
    def _split(self, block: Node):
        ordered_values = sorted(block.entries.items(), key=lambda item: item[1])
        median_index = ordered_values // 2
        # split the block into two nodes, S′ and S, where S is the original
        # node. S retains the right half of the ordered set of values in the
        # block, so keeps the same upper bound and position in the tree
        left = Node()

        for i, entry in enumerate(ordered_values):
            node, path_length = entry
            if i < median_index:
                if i == median_index - 1:
                    # set the upper bound value of the new node to the largest
                    # path length in the left half of the ordered values
                    left.value = path_length
                left.entries[node] = path_length
            else:
                block.entries[node] = path_length

        # find the correct parent for the new node
        #
        # the tree _insert function requires that the node be positioned to
        # replace a null node. if the left node is null, we can simply pass the
        # original node S as the parent. otherwise, we need to walk the subtree
        # to the left of S (since ∀b ∈ S : b <= B[S], b > B[U] for all other U)
        if block.left is None:
            self._insert(left, block, Direction.LEFT)
        else:
            parent = block.left
            # the split node S′ will always have the greatest upper bound to
            # the left of S, so simply walk right until we find a null node
            while parent.right is not None:
                parent = parent.right
            self._insert(left, parent, Direction.RIGHT)

    def search(self, value: int) -> Node:
        """Returns the node with the smallest upper bound greater than value"""
        node = self.root
        while True:
            if node.left is not None and node.left.value >= value:
                node = node.left
            elif node.value >= value:
                return node
            elif node.right is not None:
                node = node.right
            else:
                return None

    def smallest(self, node: Node = None) -> Node:
        node = node or self.root
        while node.left is not None:
            node = node.left
        return node

    # Insert(a, b) To insert a key/value pair ⟨a, b⟩, we first check the
    #   existence of its key a. If a already exists, we delete original pair
    #   ⟨a, b′⟩ and insert new pair ⟨a, b⟩ only when b < b′. Then we insert
    #   ⟨a, b⟩ to D1. We first locate the appropriate block for it, which is
    #   the block with the smallest upper bound greater than or equal to b,
    #   using binary search (via the binary search tree) on the block sequence.
    #   ⟨a, b⟩ is then added to the corresponding linked list in O(1) time.
    #   Given that the number of blocks in D1 is O(max{1, N/M}), as we will
    #   establish later, the total time complexity for a single insertion is
    #   O(max{1, log(N/M)}).
    #
    #   After an insertion, the size of the block may increase, and if it
    #   exceeds the size limit M, a split operation will be triggered.
    def insert(self, key, value):
        block = self.search(value)
        if block is None:
            self.root = Node(math.inf)
        if key in block.entries and value >= block.entries[key]:
            return  # if this is a less optimal path, don't store it
        block.entries[key] = value
        if len(block.entries) > self.block_size:
            self._split(block)

    def remove(self, node: Node) -> Node:
        """Remove the given node and rebalance the tree"""
        if node is None:
            return

        direction = node.get_direction()
        parent = node.parent

        # node has 2 non-null children, swap with the next largest value, the
        # leftmost child of this node's right child, then remove it
        if node.left is not None and node.right is not None:
            next_node = self.smallest(node.right)
            if parent is None:
                self.root = next_node
            else:
                parent.set_child(direction, next_node)
            self.remove(next_node)
            return

        # node has 1 child. simply replace with its child and colour it black
        if node.left is not None or node.right is not None:
            child = node.left or node.right
            if parent is None:
                self.root = child
            else:
                parent.set_child(direction, child)
            child.colour = Colour.BLACK
            return

        # node's children are both null

        # if this is the root, reset the tree
        if parent is None:
            self.root = Node(math.inf)
            return

        # otherwise, remove the child
        parent.set_child(direction, None)
        # if the removed node is a red leaf, we don't need to do anything else
        if node.colour == Colour.RED:
            return

        distant_nephew: Node = None

        while parent is not None:
            sibling = parent.get_child(Direction(1 - direction))
            close_nephew = sibling.get_child(direction)

            if sibling.colour == Colour.RED:
                self._rotate_subtree(parent, direction)
                parent.colour = Colour.RED
                sibling.colour = Colour.BLACK
                sibling = close_nephew

                distant_nephew = sibling.get_child(Direction(1 - direction))
                if distant_nephew and distant_nephew.colour == Colour.RED:
                    break
                close_nephew = sibling.get_child(direction)
                if close_nephew and close_nephew.colour == Colour.RED:
                    break

                sibling.colour = Colour.RED
                parent.colour = Colour.BLACK
                return

            if distant_nephew and distant_nephew.colour == Colour.RED:
                break

            if close_nephew and close_nephew.colour == Colour.RED:
                break

            if parent.colour == Colour.RED:
                sibling.colour = Colour.RED
                parent.colour = Colour.BLACK
                return

            sibling.colour = Colour.RED
            node = parent

            parent = node.parent
            direction = node.get_direction()

        if parent is None:
            return

        if (distant_nephew is None or distant_nephew.colour == Colour.BLACK
                and close_nephew and close_nephew.colour == Colour.BLACK):
            self._rotate_subtree(sibling, Direction(1 - direction))
            sibling.colour = Colour.RED
            close_nephew.colour = Colour.BLACK
            distant_nephew = sibling
            sibling = close_nephew

        self._rotate_subtree(parent, direction)
        sibling.colour = parent.colour
        parent.colour = Colour.BLACK
        distant_nephew.colour = Colour.BLACK

    def _rotate_subtree(self, root: Node, direction: Direction):
        sub_parent = root.parent
        new_root = root.get_child(Direction(1 - direction))
        new_child = new_root.get_child(direction)

        root.set_child(Direction(1 - direction), new_child)

        if new_child is not None:
            new_child.parent = root

        new_root.set_child(direction, root)

        new_root.parent = sub_parent
        root.parent = new_root
        if sub_parent is not None:
            d = Direction.RIGHT if root == sub_parent.right else Direction.LEFT
            sub_parent.set_child(d, new_root)
        else:
            self.root = new_root

        return new_root

    def clear(self):
        self.root = Node(self._upper_bound)


class PathStore:

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self._upper_bound = upper_bound
        # use a deque to hold blocks so we can prepend in O(1)
        self.batch_entries = []
        self.block_tree = BlockTree(upper_bound)
        self._counter = itertools.count()

    def insert(self, key, value):
        self.block_tree.insert(key, value)

    # BatchPrepend(L) Let L denote the size of L. When L ≤ M , we simply create a
    #   new block for L and add it to the beginning of D0. Otherwise, we create
    #   O(L/M) new blocks in the beginning of D0, each containing at most ⌈M/2⌉
    #   elements. We can achieve this by repeatedly taking medians which completes
    #   in O(L log(L/M)) time.
    def batch_prepend(self, nodes: list):
        # XXX: there are a few issues with the function as described.
        #
        # First, the description above doesn't take into account that blocks
        #   need to be in sorted order, meaning insertion will actually take
        #   an additional O(log(n)) where n is the number of blocks in D0
        #
        # Needing sorted order also means that L itself needs to be sorted
        #
        # Second, we need a boundary value in order to sort L in with existing
        #   blocks. This means we need to walk each of the O(L/M) blocks and
        #   find the maximum value, which is O(L). pull remains O(M)
        #
        # Finally, L may contain elements with path smaller than another
        #   block's upper bound, but greater than its smallest entry. In this
        #   case, we'd need another O(log(n)) to peek each block, pushing the
        #   rightmost nodes from L until max(L) <= min(S) where S is the next largest block
        #
        # Given
        #
        # M = 3
        # D0 = [(2, 3), (4, 5, 6)]
        # L -> (1, 2, 3, 4, 5)
        #
        # we would want to end up with
        #
        # D0 = [(1, 2), (2, 3), (4, 4), (5, 5, 6)]
        #
        # which seems highly inefficient. It may make more sense to just use a
        # heap, in which case the time complexity becomes O(L log(n)) where n
        # is the heap size
        #
        # for entry u ∈ L -> O(L)
        #   find the smallest bound b ∈ D0 >= u -> O(log(n))
        #   push entry to that block, then split if > M -> O(log(n))
        #
        # VS heap impl,
        #
        # for entry u ∈ L -> O(L)
        #   minheap insert -> O(log n))
        #
        # popping entries in pull() is slower (O(M)) but shouldn't affect the
        # overall runtime complexity of the function
        #
        # Given the issues above, just implement as a min-heap for now
        #
        for node in nodes:
            entry = (node[0], next(self._counter), node)
            heapq.heappush(self.batch_entries, entry)

    # Delete(a, b) To delete the key/value pair ⟨a, b⟩, we remove it directly from
    #   the linked list, which can be done in O(1) time (?). Note that it’s
    #   unnecessary to update the upper bounds of blocks after a deletion. However,
    #   if a block in D1 becomes empty after deletion, we need to remove its upper
    #   bound in the binary search tree in O(max{1, log(N/M)}) time. Since Insert
    #   takes O(max{1, log(N/M)}) time for D1, deletion time will be amortized to
    #   insertion time with no extra cost.
    def delete(self, key, value):
        # XXX: not sure how we're supposed to remove an element from an ordered
        #   list in O(1) time. Search of the tree takes O(log(N/M)) to find the
        #   node with the smallest B >= b, finding the element in the linked
        #   list runs in O(M), and THEN deletion is O(1). So a call to delete
        #   is O(M log(N/M)) without some kind of indexed lookup.
        #
        # Also unsure why we're using a linked list here in the first place.
        #   When we Pull(), we'll always be popping a full block from the tree.
        #
        # What if we instead maintain a map containing all ⟨a, b⟩ pairs, then
        #   store an ordered list of keys in each block? We would still need to
        #   remove the key from the list on delete... we could have a map of
        #   B[i] values such that { B[i] -> {a -> b} : b <= B[i] }. Then we
        #   only need to traverse the tree to find B[i] and deletion is O(1).
        #   The issue here is that we lose ordering, but the Split() operation
        #   already takes O(M) so that shouldn't really be an issue.
        #
        # Still, this is O(log(N/M)) as we always need a search to find B[i]...
        block = self.block_tree.search(value)
        block.entries.pop(key)
        if not block.entries:
            self.block_tree.remove(block)

    # Pull() To retrieve the smallest M values from D0 ∪ D1, we collect a
    #   sufficient prefix of blocks from D0 and D1 separately, denoted as S′0 and
    #   S′1, respectively. That is, in D0 (D1) we start from the first block and
    #   stop collecting as long as we have collected all the remaining elements or
    #   the number of collected elements in S′0 (S′1) has reached M. If S′0 ∪ S′1
    #   contains no more than M elements, it must contain all blocks in D0 ∪ D1, so
    #   we return all elements in S′0 ∪ S′1 as S′ and set x to the upper bound B,
    #   and the time needed is O(|S′|). Otherwise, we want to make |S′| = M, and
    #   because the block sizes are kept at most M, the collecting process takes
    #   O(M) time.
    #
    #   Now we know the smallest M elements must be contained in S′0 ∪ S′1 and can
    #   be identified from S′0 ∪ S′1 as S′ in O(M) time. Then we delete elements in
    #   S′ from D0 and D1, whose running time is amortized to insertion time. Also
    #   set returned value x to the smallest remaining value in D0 ∪ D1, which can
    #   also be found in O(M) time.
    def pull(self):
        s0 = set()
        s1 = set()

        while self.batch_entries and len(s0) < self.block_size:  # O(M)
            _, _, entry = heapq.heappop(self.batch_entries)
            s0.add(entry)

        while not self.block_tree.is_empty() and len(s1) < self.block_size:
            # pop the block with the smallest upper bound
            block = self.block_tree.remove(self.block_tree.smallest())
            s1 |= {(v, k) for k, v in block.entries}  # O(max{M, log(n)})

        s = s0 | s1

        if len(s) > self.block_size:
            # XXX: sorted based on lexicographical ordering of (b, a) tuples
            s = set(sorted(list(s))[:self.block_size])  # O(M)
            # push back any values that aren't being returned
            # by taking the difference of si and s
            self.batch_prepend(list(s0 - s))  # O(M log(n))
            [self.insert(k, v) for v, k in s1 - s]  # O(M log(n))

        d0_min = self.batch_entries[0][0]
        d1_min = min(self.block_tree.smallest().entries.values())
        lower_bound = min(d0_min, d1_min)
        return lower_bound, s
