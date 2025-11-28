import bisect
import itertools
import math
from dataclasses import dataclass
from dataclasses import field


@dataclass
class PaintersPartitionSolver:
    xs: list[int]  # list of paintings
    k: int  # number of painters / partitions

    # various properties of xs, cached after being computed once
    _min_xs: int = field(init=False, default=0)
    _max_xs: int = field(init=False, default=0)
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
    # lists of length (k+1) for the bounds of all endpoints for all partitions
    # the 1st partition always starts at the start of the list, so both lists always start with 0
    # the k-th partition always ends at the end of the list, so both lists always end with len(xs)
    # or maybe make it length k and just set the last elem to len(xs) to make the code easier to write
    _partition_boundary_lo: list[int] = field(init=False, default_factory=list)
    _partition_boundary_hi: list[int] = field(init=False, default_factory=list)

    # for completeness - we store zeroes in a separate array for reconstruction at the end
    _zero_indices: list[int] = field(init=False, default_factory=list)

    def __post_init__(self):
        """
        precompute the cumulative sum, max, and len
        (total sum is the last elem of cumulative sum)
        overall O(N) runtime if k < N
        """
        xs_without_zeroes = []  # xs without any zeroes

        # 1st O(N) pass: loop to precompute all the properties of xs
        # note that it would be much faster to use itertools.accumulate, but this demonstrates it can be a single loop
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

        # sanity check, not part of algorithm
        if xs_without_zeroes:
            assert self._min_xs == min(xs_without_zeroes)
            assert self._max_xs == max(xs_without_zeroes)
            assert self._sum_xs == sum(self.xs) == sum(xs_without_zeroes)
            assert self._cumulative_sum == list(itertools.accumulate(xs_without_zeroes))
        assert xs_without_zeroes == [x for x in self.xs if x]
        assert len(self._zero_indices) == len(self.xs) - len(xs_without_zeroes)

        # reassign self.xs
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
        assert self._min_partition_size > 0
        self._max_partition_size = min(
            self._sum_xs,
            int(math.ceil(self._sum_xs / self.k)) + self._max_xs,
        )
        assert self._max_partition_size >= self._min_partition_size
        # print(f'{self._min_partition_size=}')
        # print(f'{self._max_partition_size=}')

        # 2nd O(N) pass: precompute jump tables
        pointer_min_left = 0
        pointer_max_left = 0
        for i in range(len(self.xs)):
            while self.range_sum(pointer_min_left, i) > self._min_partition_size:
                self._min_partition_jump_table.append(i - 1)
                pointer_min_left += 1
            while self.range_sum(pointer_max_left, i) > self._max_partition_size:
                self._max_partition_jump_table.append(i - 1)
                pointer_max_left += 1
            self._min_partition_reverse_jump_table.append(pointer_min_left)
            self._max_partition_reverse_jump_table.append(pointer_max_left)

        # there's no further to jump, so just jump to the end
        assert len(self._min_partition_jump_table) == pointer_min_left
        assert len(self._max_partition_jump_table) == pointer_max_left
        self._min_partition_jump_table.extend([len(self.xs) - 1] * (len(self.xs) - pointer_min_left))
        self._max_partition_jump_table.extend([len(self.xs) - 1] * (len(self.xs) - pointer_max_left))
        # print(f'{self._min_partition_jump_table=}')
        # print(f'{self._max_partition_jump_table=}')

        # sanity check the length
        assert len(self._min_partition_jump_table) == len(self.xs)
        assert len(self._max_partition_jump_table) == len(self.xs)
        assert len(self._min_partition_reverse_jump_table) == len(self.xs)
        assert len(self._max_partition_reverse_jump_table) == len(self.xs)
        # print(f'{self._min_partition_reverse_jump_table=}')
        # print(f'{self._max_partition_reverse_jump_table=}')

        # sanity check the endpoints, which should point to themselves
        assert self._min_partition_jump_table[-1] == len(self.xs) - 1
        assert self._max_partition_jump_table[-1] == len(self.xs) - 1
        assert self._min_partition_reverse_jump_table[0] == 0
        assert self._max_partition_reverse_jump_table[0] == 0

        # no other item should point to itself, which is a partition of zero size
        assert all(self._min_partition_jump_table[i] >= i for i in range(len(self.xs) - 1))
        assert all(self._max_partition_jump_table[i] >= i for i in range(len(self.xs) - 1))
        assert all(self._min_partition_reverse_jump_table[i] <= i for i in range(1, len(self.xs)))
        assert all(self._max_partition_reverse_jump_table[i] <= i for i in range(1, len(self.xs)))

        # 3rd O(K) pass: build partition boundary lookup tables
        # this could totally have been merged into the above loop but is separate for improved readability
        self._partition_boundary_lo.append(0)  # the first partition always starts at 0
        self._partition_boundary_hi.append(0)
        pointer_min_left = 0
        pointer_max_left = 0
        for _ in range(self.k):
            pointer_min_left = self._min_partition_jump_table[pointer_min_left]
            self._partition_boundary_lo.append(pointer_min_left)
            if pointer_min_left < len(self.xs) - 1:
                pointer_min_left += 1
            pointer_max_left = self._max_partition_jump_table[pointer_max_left]
            self._partition_boundary_hi.append(pointer_max_left)
            if pointer_max_left < len(self.xs) - 1:
                pointer_max_left += 1
        self._partition_boundary_lo[-1] = len(self.xs) - 1  # the last partition always ends at the end

        # sanity check the boundaries
        assert len(self._partition_boundary_lo) == self.k + 1
        assert len(self._partition_boundary_hi) == self.k + 1
        assert self._partition_boundary_lo[0] == 0
        assert self._partition_boundary_hi[0] == 0
        assert self._partition_boundary_lo[-1] == len(self.xs) - 1
        assert self._partition_boundary_hi[-1] == len(self.xs) - 1  # this must have reached the end
        assert all((hi >= lo) for hi, lo in zip(self._partition_boundary_hi, self._partition_boundary_lo))
        # print(f'{self._partition_boundary_lo=}')
        # print(f'{self._partition_boundary_hi=}')

        # TODO: this somehow creates incorrect bounds
        # 4th O(N) pass: tighten partition bounds by looking in reverse
        # this could totally have been merged into the above loop but is separate for improved readability
        pointer_min_right = len(self.xs) - 1
        pointer_max_right = len(self.xs) - 1
        for _k in range(self.k - 1, 0, -1):
            assert self.range_sum(self._min_partition_reverse_jump_table[pointer_min_right], pointer_min_right
                                  ) <= self._min_partition_size
            assert self.range_sum(self._max_partition_reverse_jump_table[pointer_max_right], pointer_max_right
                                  ) <= self._max_partition_size
            pointer_min_right = self._min_partition_reverse_jump_table[pointer_min_right]
            self._partition_boundary_hi[_k] = min(self._partition_boundary_hi[_k], pointer_min_right)
            if pointer_min_left > 0:
                pointer_min_left -= 1
            pointer_max_right = self._max_partition_reverse_jump_table[pointer_max_right]
            self._partition_boundary_lo[_k] = max(self._partition_boundary_lo[_k], pointer_max_right)
            if pointer_max_right > 0:
                pointer_max_right -= 1
            # because this bound is so strong it can accidentally push past lo, so if it happens, then just don't
            # moving the bound back means the last painter is being allocated less
            # would be more optimal to calculate all bounds first, then do one more pass to push it back towards hi, so the strongest bound propagates further 
            self._partition_boundary_hi[_k] = max(self._partition_boundary_hi[_k], self._partition_boundary_lo[_k])

        assert self._max_partition_reverse_jump_table[pointer_max_right] == 0  # this must reach 0 by next step
        assert all((hi >= lo) for hi, lo in zip(self._partition_boundary_hi, self._partition_boundary_lo))
        # print(f'{self._partition_boundary_lo=}')
        # print(f'{self._partition_boundary_hi=}')


        # TODO: linked list of which partitions need to be checked (i.e. which boundaries are ambiguous)
        # maybe use a dict of int->int as pointers, and it probably needs to be doubly linked
        # can probably safely ignore the GC and just not remove old nodes

    def range_sum(self, start: int, end: int) -> int:
        # O(1) lookup via cumulative sum
        # includes start and end
        assert start >= 0
        assert end >= start
        assert len(self._cumulative_sum) >= end + 1

        _range_sum = self._cumulative_sum[end]
        if start > 0:
            _range_sum -= self._cumulative_sum[start - 1]
        return _range_sum

    def test_partition(self, partition_size: int) -> bool:  # tuple[int, int]:
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
        # gemini suggested
        start_idx = 0
        n = len(self.xs)

        for _k in range(self.k):
            # get the bounds for the END of this k-th partition
            end_idx_lo = self._min_partition_jump_table[start_idx]
            end_idx_hi = self._max_partition_jump_table[start_idx]
            # print(f'{self._min_partition_jump_table[start_idx]=}')
            # print(f'{self._max_partition_jump_table[start_idx]=}')
            # print(f'{self._partition_boundary_lo[_k + 1]=}')
            # print(f'{self._partition_boundary_hi[_k + 1]=}')

            # # ~~partition boundary is not working, so some invariant somewhere died~~
            # # note that the first elem of partition_boundary is the start of the 0-th partition
            end_idx_lo = max(self._min_partition_jump_table[start_idx], self._partition_boundary_lo[_k + 1])
            end_idx_hi = min(self._max_partition_jump_table[start_idx], self._partition_boundary_hi[_k + 1])
            # so it turns out that this constraint can combine with the above one to produce a range where no solution is possible
            # but bisect will still output whatever even if the
            # so the correctness dies
            # i added an o(1) check to check below so if bisect does something silly we just exit false

            # We want to find the largest index 'p' in [lo_idx, hi_idx] such that range_sum(current_idx, p) <= partition_size.
            # range_sum(current_idx, p) = cumsum[p] - (cumsum[current_idx-1])
            # So: cumsum[p] <= partition_size + (cumsum[current_idx-1])

            prev_sum = self.range_sum(0, start_idx - 1) if _k else 0
            target = prev_sum + partition_size

            # bisect_right returns insertion point.
            # We search in _cumulative_sum within the bounds [lo_idx, hi_idx + 1]
            # (Note: +1 because bisect range is lo, hi exclusive at end)
            next_idx = bisect.bisect_right(self._cumulative_sum, target, lo=end_idx_lo, hi=min(end_idx_hi + 1, n)) - 1
            if self.range_sum(start_idx, next_idx) > partition_size:
                return False

            # print(f'{_k=}, {end_idx_lo=} {end_idx_hi=} {target=} {next_idx=}')

            # If we reached the end of the array, we are done
            if next_idx >= n - 1:
                # print(f'{partition_size=} {_k=}, {end_idx_lo=} {end_idx_hi=} {target=} {next_idx=} True')
                return True

            start_idx = next_idx + 1

            # If current_idx went past the bounds, something is wrong or finished (handled above)
            if start_idx >= n:
                # print(f'{partition_size=} {_k=}, {end_idx_lo=} {end_idx_hi=} {target=} {next_idx=} True 2')
                return True

        # print(f'{partition_size=} False')
        return False

    def solve_partition(self) -> int:
        """
        tests partitions between min and max partition
        tries to find the partition with zero remainder
        if not, then the smallest negative remainder (i.e. the smallest possible partition that works

        :return:
        """
        # a simple implementation would be to bisect over min and max
        # but because we can optimize further when the partition is smaller but still too big
        # it requires (this sentence was cut off...? well wtv refer to readme)

        # gemini suggested this simple implementation, which does run correctly
        if not self.xs: return 0
        search_space = range(self._min_partition_size, self._max_partition_size + 1)
        idx = bisect.bisect_left(search_space, True, key=self.test_partition)
        return search_space[idx]


def painter(xs, k):
    # O(len(xs) * log(sum(xs))
    def is_possible(time_limit):
        count, current_sum = 1, 0
        for length in xs:
            if current_sum + length > time_limit:
                count += 1
                current_sum = length
            else:
                current_sum += length
        return count <= k

    values = range(max(xs), sum(xs) + 1)
    index = bisect.bisect_left(values, True, key=is_possible)

    return values[index]


if __name__ == '__main__':
    import random
    import time

    for attempt in range(trials := 100):
        xs = [random.randint(1, 1_000_000_000) for _ in range(random.randint(1, 1000_000))]
        k = random.randint(1, 1_000)
        print(f'[{attempt + 1}/{trials}]', len(xs), k)  # , xs)
        t = time.time()
        solver = PaintersPartitionSolver(xs=xs, k=k)
        print('precompute', time.time() - t)
        answer = solver.solve_partition()
        print('solver', time.time() - t)
        t = time.time()
        ans2 = painter(xs, k)
        print('dp', time.time() - t)
        print(answer)
        assert ans2 == answer, ans2
        print('-' * 100)
