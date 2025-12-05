from collections import OrderedDict
from typing import Iterable


class OldDoublyLinkedList:
    """
    not used in favor of a better, more python-native solution
    """

    def __init__(self, initial_values: Iterable[int] = ()):
        self.forward_links: dict[int | None, int | None] = {}
        self.backward_links: dict[int | None, int | None] = {}

        # Initialize logic
        previous_element = element = None
        for element in initial_values:
            if element in self.backward_links:
                raise ValueError(f"Duplicate value {element} found. Values must be unique.")

            # Link previous -> current
            self.forward_links[previous_element] = element
            # Link current -> previous
            self.backward_links[element] = previous_element
            previous_element = element

        # set tail pointer
        self.backward_links[None] = previous_element
        self.forward_links[previous_element] = None

    @property
    def head(self) -> int:
        val = self.forward_links[None]
        if val is None:
            raise StopIteration("List is empty")  # Handle empty list case explicitly
        return val

    @property
    def tail(self) -> int:
        val = self.backward_links[None]
        if val is None:
            raise StopIteration("List is empty")
        return val

    def append(self, element: int):
        if element in self.forward_links:  # Check if value exists anywhere
            raise ValueError(f"Value {element} already exists.")

        old_tail: int | None = self.backward_links[None]
        self.forward_links[old_tail] = element
        self.forward_links[element] = None
        self.backward_links[None] = element
        self.backward_links[element] = old_tail

    def prepend(self, element: int):
        if element in self.forward_links:
            raise ValueError(f"Value {element} already exists.")

        old_head: int | None = self.forward_links[None]
        self.backward_links[old_head] = element
        self.backward_links[element] = None
        self.forward_links[None] = element
        self.forward_links[element] = old_head

    def pop(self, element: int):
        # We check forward_links specifically, but valid nodes should be in both
        if element not in self.forward_links:
            raise KeyError(f"Element {element} not found.")

        # Identify neighbors
        # Use .get() to handle None boundaries safely
        prev_node = self.backward_links[element]
        next_node = self.forward_links[element]
        self.forward_links[prev_node] = next_node
        self.backward_links[next_node] = prev_node

        # cleanup
        del self.forward_links[element]
        del self.backward_links[element]

    def __iter__(self):
        item = None
        while (item := self.forward_links[item]) is not None:
            yield item

    def __reversed__(self):
        item = None
        while (item := self.backward_links[item]) is not None:
            yield item

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self)!r})'


class DoublyLinkedList:
    """
    Production-ready implementation backed by C-optimized OrderedDict.
    Maintains O(1) performance for head/tail/middle operations.
    """

    def __init__(self, initial_values: Iterable[int] = ()):
        # We use a loop here to enforce the "No Duplicates" rule
        # that OrderedDict.fromkeys would ignore (and take the first item's position)
        self._data = OrderedDict()
        for v in initial_values:
            if v in self._data:
                raise ValueError(f"Duplicate value {v} found.")
            self._data[v] = None

    @property
    def head(self):
        try:
            return next(iter(self._data))
        except StopIteration:
            raise StopIteration("List is empty")

    @property
    def tail(self):
        try:
            return next(reversed(self._data))
        except StopIteration:
            raise StopIteration("List is empty")

    def append(self, item):
        if item in self._data:
            raise ValueError(f"Value {item} already exists.")
        self._data[item] = None

    def prepend(self, item):
        if item in self._data:
            raise ValueError(f"Value {item} already exists.")
        self._data[item] = None
        self._data.move_to_end(item, last=False)

    def pop(self, item):
        del self._data[item]

    def __iter__(self):
        yield from self._data

    def __reversed__(self):
        return reversed(self._data)

    def __repr__(self):
        return f'{self.__class__.__name__}({list(self)!r})'
