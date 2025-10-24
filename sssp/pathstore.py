import itertools
import math

from .rbtree import Node, RedBlackTree, Direction


class Block:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = {}

    def size(self):
        return len(self.entries)

    def __len__(self):
        return self.size()


class TreeBlock(Block, Node):

    def __init__(self, upper_bound: float):
        super().__init__(upper_bound)

    def replace(self, node: "TreeBlock"):
        super().replace(node)
        self.entries = node.entries


class ListNode:

    def __init__(self, value):
        self.value = value
        self.prev: ListNode = None
        self.next_: ListNode = None


class ListBlock(Block, ListNode):

    def __init__(self, upper_bound: float):
        super().__init__(upper_bound)


class BlockList:

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self._upper_bound = upper_bound
        self.head = ListBlock(upper_bound)
        self.tail = self.head

    def is_empty(self):
        return not self.head.entries

    def minimum(self) -> float:
        if self.is_empty():
            return math.inf
        return min(self.head.entries.values())

    def pull(self) -> set:
        s = set()
        while not self.is_empty() and len(s) < self.block_size:  # O(M)
            block = self.popleft()
            s |= set(block.entries.items())
        return s

    def pushleft(self, block: ListBlock):
        self.head.prev = block
        block.next_ = self.head
        self.head = block

    def popleft(self) -> ListBlock:
        popped = self.head
        self.head = self.head.next_
        if self.head is None:
            self._reset()
        self.head.prev = None
        return popped

    def _split(self, block: ListBlock):
        ordered = sorted(block.entries.items(), key=lambda item: item[1])
        median_index = len(ordered) // 2
        prev = ListBlock(ordered[median_index - 1][1])

        for i in range(median_index):
            node, path_length = ordered[i]
            prev.entries[node] = path_length
            block.entries.pop(node)

        prev.prev = block.prev
        prev.next_ = block
        block.prev = prev

        if prev.prev is not None:
            prev.prev.next_ = prev

        if block is self.head:
            self.head = prev

    def insert(self, key, value):
        larger_block = self.tail
        block = self.tail.prev

        # XXX: would be faster to bisect here
        while block is not None:
            if value > block.value:
                break
            larger_block = block
            block = larger_block.prev
        larger_block.entries[key] = value

        if len(larger_block.entries) > self.block_size:
            self._split(larger_block)

    def _reset(self):
        self.head = ListBlock(self._upper_bound)
        self.tail = self.head


class BlockTree(RedBlackTree):

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self._upper_bound = upper_bound
        self.root = TreeBlock(upper_bound)

    def is_empty(self):
        return not self.root.entries

    def minimum(self) -> float:
        if self.is_empty():
            return math.inf
        return min(self.root.entries.values())

    def pull(self) -> set:
        s = set()
        while not self.is_empty() and len(s) < self.block_size:  # O(M)
            block = self.smallest()
            self.remove(block)
            s |= set(block.entries.items())
        return s

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
    def _split(self, block: TreeBlock):
        ordered_values = sorted(block.entries.items(), key=lambda item: item[1])
        median_index = len(ordered_values) // 2
        # split the block into two nodes, S′ and S, where S is the original
        # node. S retains the right half of the ordered set of values in the
        # block, so keeps the same upper bound and position in the tree.
        # set the upper bound value of the new node to the largest
        # path length in the left half of the ordered values
        left = TreeBlock(ordered_values[median_index - 1][1])

        for i in range(median_index):
            node, path_length = ordered_values[i]
            left.entries[node] = path_length
            block.entries.pop(node)

        # find the correct parent for the new node
        #
        # the tree insert function requires that the node be positioned to
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

    def search(self, value: int) -> TreeBlock:
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
        # XXX: change this function name; currently breaks Liskov substitution
        block = self.search(value)
        if block is None:
            self.clear()
            block = self.root
        if key in block.entries and value >= block.entries[key]:
            return  # if this is a less optimal path, don't store it
        block.entries[key] = value
        if block.size() > self.block_size:
            self._split(block)

    def smallest(self, node: TreeBlock = None) -> TreeBlock:
        return super().smallest(node)

    def clear(self):
        self.root = TreeBlock(self._upper_bound)

    def pprint(self, node: TreeBlock, depth=0):
        if node is None:
            return "\t" * depth + "|_ null\n"
        # recursively draw a tree
        direction = node.get_direction()
        return ("\t" * depth + f"|_ {direction.name} | {node.value}: {node.colour} | {node.entries}\n"
                + self.pprint(node.left, depth + 1)
                + self.pprint(node.right, depth + 1))


class PathStore:

    def __init__(self, upper_bound: float, block_size: int):
        self.block_size = block_size
        self.batch_entries = BlockList(upper_bound, block_size)
        self.block_tree = BlockTree(upper_bound, block_size)
        self._counter = itertools.count()
        self._removed_entries = {}

    def is_empty(self):
        return self.batch_entries.is_empty() and self.block_tree.is_empty()

    def insert(self, key, value):
        self.block_tree.insert(key, value)

    # BatchPrepend(S) Let L denote the size of S. When L ≤ M , we simply create a
    #   new block for S and add it to the beginning of D0. Otherwise, we create
    #   O(L/M) new blocks in the beginning of D0, each containing at most ⌈M/2⌉
    #   elements. We can achieve this by repeatedly taking medians which completes
    #   in O(L log(L/M)) time.
    def batch_prepend(self, nodes: list):
        if not nodes:
            return

        smallest_in_head = math.inf
        largest_value = sorted(nodes)[-1][0]

        if not self.batch_entries.is_empty():
            head = self.batch_entries.head
            smallest_in_head = sorted(head.entries.values())[0]

        if len(nodes) <= self.block_size and largest_value <= smallest_in_head:
            block = ListBlock(largest_value)
            block.entries = {k: v for k, v in nodes}
            self.batch_entries.pushleft(block)
        else:
            for k, v in nodes:
                self.batch_entries.insert(k, v)

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
        if block.size() == 0:
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
        s0 = self.batch_entries.pull()
        s1 = self.block_tree.pull()

        s = s0 | s1

        if len(s) > self.block_size:
            s = set(sorted(list(s), key=lambda x: x[1])[:self.block_size])  # O(M)
            # push back any values that aren't being returned
            # by taking the difference of si and s
            self.batch_prepend(list(s0 - s))  # O(M log(n))
            [self.insert(k, v) for k, v in s1 - s]  # O(M log(n))

        x = min(self.batch_entries.minimum(), self.block_tree.minimum())
        return x, {k for k, _ in s}
