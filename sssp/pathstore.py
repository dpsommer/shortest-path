import heapq
import itertools
import math
from bisect import bisect_left

from .rbtree import Node, RedBlackTree, Direction


class Block(Node):

    def __init__(self, upper_bound: float):
        super().__init__(upper_bound)
        self.entries = {}


class BlockTree(RedBlackTree):

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self._upper_bound = upper_bound
        self.root = Block(upper_bound)
        self._node_count = 1

    def is_empty(self):
        return not self.root.entries

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
    def _split(self, block: Block):
        ordered_values = sorted(block.entries.items(), key=lambda item: item[1])
        median_index = len(ordered_values) // 2
        # split the block into two nodes, S′ and S, where S is the original
        # node. S retains the right half of the ordered set of values in the
        # block, so keeps the same upper bound and position in the tree.
        # set the upper bound value of the new node to the largest
        # path length in the left half of the ordered values
        left = Block(ordered_values[median_index - 1][1])

        for i in range(median_index):
            node, path_length = ordered_values[i]
            left.entries[node] = path_length
            block.entries.pop(node)

        # find the correct parent for the new node
        #
        # the tree _insert function requires that the node be positioned to
        # replace a null node. if the left node is null, we can simply pass the
        # original node S as the parent. otherwise, we need to walk the subtree
        # to the left of S (since ∀b ∈ S : b <= B[S], b > B[U] for all other U)
        if block.left is None:
            super().insert(left, block, Direction.LEFT)
        else:
            parent = block.left
            # the split node S′ will always have the greatest upper bound to
            # the left of S, so simply walk right until we find a null node
            while parent.right is not None:
                parent = parent.right
            super().insert(left, parent, Direction.RIGHT)

    def search(self, value: int) -> Block:
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
            self.root = Block(math.inf)
        if key in block.entries and value >= block.entries[key]:
            return  # if this is a less optimal path, don't store it
        block.entries[key] = value
        if len(block.entries) > self.block_size:
            self._split(block)

    def clear(self):
        self._node_count = 0
        self.root = Node(self._upper_bound)


class PathStore:

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self.batch_entries = []
        self.block_tree = BlockTree(upper_bound, block_size)
        self._counter = itertools.count()
        self._removed_entries = {}

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
        #   rightmost nodes from L until max(L) <= min(S) where S is the next
        #   largest block
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

    def _batch_entry_index(self, key, value) -> int | None:
        i = bisect_left(self.batch_entries, value, key=lambda x: x[0])
        for i in range(i, len(self.batch_entries)):
            v, k = self.batch_entries[i][2]
            if v != value:
                return None # k,v pair doesn't exist in D0 either
            if k == key:
                return i

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
        v = block.entries.pop(key, None)
        if len(block.entries) == 0:
            self.block_tree.remove(block)
        if v is None:  # remove pair from D0
            # lazy delete so we don't need to spend time maintaining the heap
            # invariant or searching for the k/v pair in the heap
            self._removed_entries[key] = value

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
            v, k = entry
            if self._removed_entries.get(k) == v:
                continue
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
