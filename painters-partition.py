from dataclasses import dataclass
from functools import cached_property


@dataclass
class PaintersPartition:
    xs: list[int]  # list of paintings
    k: int  # number of painters / partitions

    @cached_property
    def cumulative_sum(self) -> list[int]:
        # o(n) preprocessing
        return [s for s in [0] for x in self.xs for s in [s + x]]

    @cached_property
    def max_xs(self) -> int:
        # o(n) preprocessing
        return max(self.xs)

    @cached_property
    def sum_xs(self) -> int:
        # o(1) by using cumulative_sum
        return self.cumulative_sum[-1]

    def range_sum(self, start: int, end: int) -> int:
        # o(1) using cumulative_sum
        assert end > start or end < 0 and (len(self.xs) + end) > start
        return self.cumulative_sum[end] - self.cumulative_sum[start]

    # min and max partition size
    # used in outer binary search loop but does not update during the search
    min_partition_size: int | None = None
    max_partition_size: int | None = None

    # lists of length n
    # at each position, what's the position of the board at most partition_size from here
    # built only once and does not update as min and max partition sizes update during the outer search
    # only supports moving from the left
    min_partition_lookup: list[int] | None = None
    max_partition_lookup: list[int] | None = None
    # TODO: is it better practice to use tuples (i.e., [(lo,hi),...] -> `list[tuple[int,int]]`) instead?

    # (optional optimization)
    # lists of length (k-1)
    # the 1st partition always starts at the start of the list
    # the k-th partition always ends at the end of the list
    # or maybe make it length k and just set the last elem to len(xs) to make the code easier to write
    partition_boundary_lo: list[int] | None = None
    partition_boundary_hi: list[int] | None = None

    def __post_init__(self):
        # TODO: 1st pass pre-processing step to find the min, max, cumsum, sum
        ...
        # TODO: calculate min_partition and max_partition
        ...
        # TODO: 2nd pass pre-processing step to index partitions
        ...
        # TODO: optimization: build lo and hi by using min_partition forwards and min_partition+1 in reverse
        ...

    def test_partition(self, partition_size: int) -> tuple[int, int]:
        """
        if too small, returns the remainder that couldn't fit
        if too big, returns the excess space
        returns zero if and only if the last partition is completely filled

        also returns the largest partition size

        :param partition_size:
        :return:
        """
        ...

    def solve_partition(self) -> int:
        """
        tests partitions between min and max partition
        tries to find the partition with zero remainder
        if not, then the smallest negative remainder (i.e. the smallest possible partition that works

        :return:
        """
        # a simple implementation would be to bisect over min and max
        # but because we can optimize further when the partition is smaller but still too big
        # it requires
        ...
