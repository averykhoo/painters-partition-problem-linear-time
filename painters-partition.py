from dataclasses import dataclass


@dataclass
class PaintersPartition:
    xs: list[int]  # list of paintings
    k: int  # number of painters / partitions

    xs_cumulative: list[int] | None = None
    x_max: int | None = None

    min_partition: int | None = None
    max_partition: int | None = None

    min_partition_lookup: list[int] | None = None
    max_partition_lookup: list[int] | None = None

    optimization_partition_lo: list[int] | None = None
    optimization_partition_hi: list[int] | None = None

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
