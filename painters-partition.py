import math
from dataclasses import dataclass
from dataclasses import field


@dataclass
class PaintersPartition:
    xs: list[int]  # list of paintings
    k: int  # number of painters / partitions

    # @cached_property
    # def cumulative_sum(self) -> list[int]:
    #     # o(n) preprocessing
    #     return [s for s in [0] for x in self.xs for s in [s + x]]
    # @cached_property
    # def max_xs(self) -> int:
    #     # o(n) preprocessing
    #     return max(self.xs)
    # @cached_property
    # def sum_xs(self) -> int:
    #     # o(1) by using cumulative_sum
    #     return self.cumulative_sum[-1]

    # various properties of xs, cached after being computed once
    _min_xs: int = field(init=False, default=0)
    _max_xs: int = field(init=False, default=0)
    _len_xs: int = field(init=False, default=0)
    _sum_xs: int = field(init=False, default=0)

    # the cumulative sum of xs, has the same length as xs
    _cumulative_sum: list[int] = field(init=False, default_factory=list)

    # min and max partition size
    # used in outer binary search loop but does not update during the search
    _min_partition_size: int = field(init=False, default=0)
    _max_partition_size: int = field(init=False, default=0)

    # lists of length n
    # at each position, what's the position of the board at most partition_size from here
    # built only once and does not update as min and max partition sizes update during the outer search
    # left to right
    _min_partition_jump_table: list[int] = field(init=False, default_factory=list)
    _max_partition_jump_table: list[int] = field(init=False, default_factory=list)
    # right to left
    _min_partition_reverse_jump_table: list[int] = field(init=False, default_factory=list)
    _max_partition_reverse_jump_table: list[int] = field(init=False, default_factory=list)

    # (optional optimization)
    # lists of length (k-1)
    # the 1st partition always starts at the start of the list
    # the k-th partition always ends at the end of the list
    # or maybe make it length k and just set the last elem to len(xs) to make the code easier to write
    _partition_boundary_lo: list[int] = field(init=False, default_factory=list)
    _partition_boundary_hi: list[int] = field(init=False, default_factory=list)

    # for completeness - we store zeroes in a separate array for reconstruction at the end
    _zero_indices: list[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        xs_without_zeroes = []  # xs without any zeroes

        # 1st O(N) pass: loop to precompute all the properties of xs
        for i, x in enumerate(self.xs):
            if x < 0:
                raise ValueError(f'found invalid value {x} in xs, which must be >=0')
            if x == 0:
                self._zero_indices.append(i)
                continue
            self._min_xs = min(self._min_xs or x, x)
            self._max_xs = max(self._max_xs or x, x)
            self._sum_xs += x
            self._cumulative_sum.append(self._sum_xs)
            xs_without_zeroes.append(x)
        self.xs = xs_without_zeroes

        # early exit if the list was empty
        if not self.xs:
            return

        # early exit if no workers exist to do work
        if self.k <= 0:
            raise ValueError(f'found invalid value {self.k} for `k`, which must be >0')

        # calculation of min and max partition size
        self._min_partition_size = max(
            self._max_xs,
            int(math.ceil(self._sum_xs / self.k)),
        )
        self._max_partition_size = min(
            self._sum_xs,
            int(math.ceil(self._sum_xs / self.k)) + self._max_xs,
        )
        assert 0 < self._max_xs <= self._min_partition_size <= self._max_partition_size <= self._sum_xs

        # 2nd O(N) pass: precompute jump tables
        pointer_min_left = 0
        pointer_max_left = 0
        for i in range(len(self.xs)):
            while self._cumulative_sum[i] - self._cumulative_sum[pointer_min_left] > self._min_partition_size:
                self._min_partition_jump_table.append(i - 1)
                pointer_min_left += 1
            while self._cumulative_sum[i] - self._cumulative_sum[pointer_max_left] > self._max_partition_size:
                self._max_partition_jump_table.append(i - 1)
                pointer_max_left += 1
            self._min_partition_reverse_jump_table.append(pointer_min_left)
            self._max_partition_reverse_jump_table.append(pointer_max_left)

        # there's no further to jump, so just jump to the end
        assert len(self._min_partition_jump_table) == pointer_min_left
        assert len(self._max_partition_jump_table) == pointer_max_left
        self._min_partition_jump_table.extend([len(self.xs) - 1] * (len(self.xs) - pointer_min_left))
        self._max_partition_jump_table.extend([len(self.xs) - 1] * (len(self.xs) - pointer_max_left))

        # 3rd O(N) pass: build partition boundary lookup tables
        # this could totally have been merged into the above loop but is separate for improved readability
        pointer_min_left = 0
        pointer_max_left = 0
        for i in range(self.k - 1):
            if self._cumulative_sum[i] - self._cumulative_sum[pointer_min_left] > self._min_partition_size:
                self._partition_boundary_lo.append(i - 1)
                pointer_max_left = i
            if self._cumulative_sum[i] - self._cumulative_sum[pointer_max_left] > self._max_partition_size:
                self._partition_boundary_hi.append(i - 1)
                pointer_max_left = i

    def range_sum(self, start: int, end: int) -> int:
        # o(1) using cumulative_sum
        assert end > start or end < 0 and (len(self.xs) + end) > start
        return self._cumulative_sum[end] - self._cumulative_sum[start]

    def test_partition(self, partition_size: int) -> tuple[int, int]:
        """
        if too small, returns the remainder that couldn't fit
        if too big, returns the excess space
        returns zero if and only if the last partition is completely filled

        also returns the largest partition size

        gemini3pro's complaint:
            Over-engineering: The standard Binary Search only needs a boolean:
            `True` (it fits in k partitions) or `False` (it needs >k).

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
