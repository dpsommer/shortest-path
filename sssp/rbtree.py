import copy
import enum
from typing import Optional


class Direction(enum.IntEnum):
    ROOT = -1
    LEFT = 0
    RIGHT = 1


class Colour(enum.Enum):
    BLACK = 0
    RED = 1


class Node:

    def __init__(self, value: float):
        self.parent: Optional[Node] = None
        self.right: Optional[Node] = None
        self.left: Optional[Node] = None
        self.colour = Colour.RED
        self.value = value

    def get_child(self, direction: Direction):
        if direction == Direction.LEFT:
            return self.left
        return self.right

    def set_child(self, direction: Direction, node: Optional["Node"]):
        if direction == Direction.LEFT:
            self.left = node
        else:
            self.right = node

    def get_direction(self) -> Direction:
        if self.parent is None:
            return Direction.ROOT
        return Direction.LEFT if self == self.parent.left else Direction.RIGHT

    def replace(self, node: "Node"):
        self.value = node.value


class RedBlackTree:

    def __init__(self):
        self.root = None

    def is_empty(self):
        return not self.root

    def insert(self, node: Node, parent: Node, direction: Direction):
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

    def search(self, value: int) -> Optional[Node]:
        """Returns the node with the smallest upper bound greater than value"""
        node = self.root
        while True:
            if node.value == value:
                return node
            elif node.left is not None and value <= node.left.value:
                node = node.left
            elif node.right is not None and value >= node.right.value:
                node = node.right
            else:
                return None

    def smallest(self, node: Node = None) -> Node:
        """Returns the leftmost leaf node in the tree"""
        node = node or self.root
        while node.left is not None:
            node = node.left
        return node

    def remove(self, node: Node):
        """Remove the given node and rebalance the tree"""
        if node is None:
            return

        direction = node.get_direction()
        parent = node.parent

        # node has 2 non-null children, swap with the next largest value, the
        # leftmost child of this node's right child, then remove it
        if node.left is not None and node.right is not None:
            next_node = self.smallest(node.right)

            clone = copy.copy(node)
            clone.replace(next_node)
            clone.colour = node.colour
            clone.left = node.left
            clone.right = node.right

            node.left.parent = clone
            node.right.parent = clone

            if parent is None:
                self.root = clone
            else:
                parent.set_child(direction, clone)

            self.remove(next_node)
            return

        # node has 1 child. replace with its child and colour it black
        if node.left is not None or node.right is not None:
            child = node.left or node.right
            if parent is None:
                self.root = child
            else:
                parent.set_child(direction, child)
            child.parent = parent
            child.colour = Colour.BLACK
            return

        # node's children are both null

        # if this is the root, reset the tree
        if parent is None:
            self.clear()
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
            distant_nephew = sibling.get_child(Direction(1 - direction))

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
                and close_nephew and close_nephew.colour == Colour.RED):
            self._rotate_subtree(sibling, Direction(1 - direction))
            sibling.colour = Colour.RED
            close_nephew.colour = Colour.BLACK
            distant_nephew = sibling
            sibling = close_nephew

        self._rotate_subtree(parent, direction)
        sibling.colour = parent.colour
        parent.colour = Colour.BLACK
        distant_nephew.colour = Colour.BLACK

    def _rotate_subtree(self, sub: Node, direction: Direction):
        sub_parent = sub.parent
        new_root = sub.get_child(Direction(1 - direction))
        new_child = new_root.get_child(direction)

        sub.set_child(Direction(1 - direction), new_child)

        if new_child is not None:
            new_child.parent = sub

        new_root.set_child(direction, sub)

        new_root.parent = sub_parent
        sub.parent = new_root
        if sub_parent is not None:
            d = Direction.RIGHT if sub == sub_parent.right else Direction.LEFT
            sub_parent.set_child(d, new_root)
        else:
            self.root = new_root

        return new_root

    def clear(self):
        self.root = None

    def height(self, node: Node) -> int:
        if node is None:
            return -1
        return 1 + max(self.height(node.left), self.height(node.right))

    def pprint(self, node: Node, depth=0):
        if node is None:
            return "\t" * depth + "|_ null\n"
        # recursively draw a tree
        direction = node.get_direction()
        return ("\t" * depth + f"|_ {direction.name} | {node.value}: {node.colour}\n"
                + self.pprint(node.left, depth + 1)
                + self.pprint(node.right, depth + 1))
