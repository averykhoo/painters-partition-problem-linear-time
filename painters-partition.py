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
    # TODO: fine-tune the boundary definition which currently is a bit fuzzy
    # note that a boundary lies immediately after the index number, i.e.:
    # 0 points at the `,` in `[0,1]`
    _partition_boundary_lo: list[int] = field(init=False, default_factory=list)
    _partition_boundary_hi: list[int] = field(init=False, default_factory=list)

    # for completeness - we store zeroes in a separate array for reconstruction at the end
    _zero_indices: list[int] = field(init=False, default_factory=list)

    def __build_cumulative_sum(self):
        """
        builds the cumulative sum array
        also caches the min, max, and sum

        the len is not cached since that's O(1) in python anyway
        but in some other languages that might be needed to maintain the overall runtime bound
        """

        xs_without_zeroes = []  # xs without any zeroes

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
            assert self._sum_xs == sum(self.xs) == sum(xs_without_zeroes) == self._cumulative_sum[-1]
            assert self._cumulative_sum == list(itertools.accumulate(xs_without_zeroes))
        assert xs_without_zeroes == [x for x in self.xs if x]
        assert len(self._zero_indices) == len(self.xs) - len(xs_without_zeroes)

        # reassign self.xs
        self.xs = xs_without_zeroes

    def __calculate_partition_size_bounds(self):
        """
        checks some conditions and calculates the smallest possible and largest necessary partition sizes
        these bounds are inclusive (and obviously closed), and if min==max then that is the only possible answer
        """

        # early exit if we have more workers than partitions
        if self.k >= len(self.xs):
            self._min_partition_size = self._max_partition_size = self._max_xs
            return

        # early exit when this special condition holds - we know the partition is exactly equal to the max
        if self.k % 2 == 1:
            if self._max_xs * (self.k - 1) >= 2 * (self._sum_xs - max(self.xs[0], self.xs[-1])) - self.k + 1:
                self._min_partition_size = self._max_partition_size = self._max_xs
                return
        else:
            if self._max_xs * self.k >= 2 * self._sum_xs - self.k:
                self._min_partition_size = self._max_partition_size = self._max_xs
                return

        # calculation of min and max partition size
        self._min_partition_size = max(
            self._max_xs,
            int(math.ceil(self._sum_xs / self.k)),
        )
        self._max_partition_size = min(
            self._sum_xs,
            int(math.ceil(self._sum_xs / self.k)) + self._max_xs,
        )

        # tighter bound in this special case
        if self._max_xs * self.k < self._sum_xs:
            self._max_partition_size = min(
                self._max_partition_size,
                int(math.ceil((self._sum_xs + (self._max_xs - 1) * (self.k - 1)) / self.k)),
            )

        assert self._max_partition_size >= self._min_partition_size > 0

    def __build_jump_tables(self):
        """
        builds the forwards and backwards jump tables based on min and max partition sizes
        """
        assert self._max_partition_size > self._min_partition_size >= self._max_xs > 0

        pointer_min_left = 0
        pointer_max_left = 0
        max_observed_partition_size = 0
        for i in range(len(self.xs)):
            while self.range_sum(pointer_min_left, i) > self._min_partition_size:
                self._min_partition_jump_table.append(i - 1)
                pointer_min_left += 1
            while (current_max_partition_size := self.range_sum(pointer_max_left, i)) > self._max_partition_size:
                self._max_partition_jump_table.append(i - 1)
                pointer_max_left += 1
            self._min_partition_reverse_jump_table.append(pointer_min_left)
            self._max_partition_reverse_jump_table.append(pointer_max_left)
            max_observed_partition_size = max(max_observed_partition_size, current_max_partition_size)

        # extend the forwards table to the end
        assert pointer_min_left == len(self._min_partition_jump_table)
        assert pointer_max_left == len(self._max_partition_jump_table)
        self._min_partition_jump_table.extend([len(self.xs) - 1] * (len(self.xs) - pointer_min_left))
        self._max_partition_jump_table.extend([len(self.xs) - 1] * (len(self.xs) - pointer_max_left))

        # sanity check and then optimize the max partition size
        assert max_observed_partition_size <= self._max_partition_size
        self._max_partition_size = max_observed_partition_size

        # sanity check the length
        assert len(self._min_partition_jump_table) == len(self.xs)
        assert len(self._max_partition_jump_table) == len(self.xs)
        assert len(self._min_partition_reverse_jump_table) == len(self.xs)
        assert len(self._max_partition_reverse_jump_table) == len(self.xs)

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

        # validation pass - run the whole thing in reverse just to make sure the reverse pass logic was right
        # remove this entire block in production
        validation_min_partition_reverse_jump_table = []
        validation_max_partition_reverse_jump_table = []
        pointer_min_right = len(self.xs) - 1
        pointer_max_right = len(self.xs) - 1
        for i in range(len(self.xs) - 1, -1, -1):  # in reverse
            while self.range_sum(i, pointer_min_right) > self._min_partition_size:
                validation_min_partition_reverse_jump_table.append(i + 1)
                pointer_min_right -= 1
            while self.range_sum(i, pointer_max_right) > self._max_partition_size:
                validation_max_partition_reverse_jump_table.append(i + 1)
                pointer_max_right -= 1
        validation_min_partition_reverse_jump_table.extend([0] * (pointer_min_right + 1))
        validation_max_partition_reverse_jump_table.extend([0] * (pointer_max_right + 1))
        validation_min_partition_reverse_jump_table.reverse()
        validation_max_partition_reverse_jump_table.reverse()
        assert self._min_partition_reverse_jump_table == validation_min_partition_reverse_jump_table
        assert self._max_partition_reverse_jump_table == validation_max_partition_reverse_jump_table

    def __build_partition_boundary_lists(self):
        """
        builds the pair of (lo, hi) partition boundary constraint lists

        this could totally have been merged into the 2nd pass code (building jump tables),
        but is kept separate for improved readability

        note that min partition size is incremented by one
        unless it succeeded in which the max:=min since we found the answer
        """

        # the first partition always starts before item 0, i.e., after item "-1"
        # this value is added for completeness, but is never actually used
        # in practice we only use the values for indexes 1 through n-1, and ignore 0 and n, since those are fixed
        self._partition_boundary_lo.append(-1)
        self._partition_boundary_hi.append(-1)
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

        # early exit if the smallest possible partition is sufficient
        # this means we've already found the answer
        if self._partition_boundary_lo[-1] == len(self.xs) - 1:
            self._max_partition_size = self._min_partition_size
            return
        self._partition_boundary_lo[-1] = len(self.xs) - 1  # the last partition always ends at the end

        # the reason min_partition_size can safely be incremented by one is because
        # at this point we know that it did not successfully partition the list
        self._min_partition_size += 1

        # (optional) if there was no space left in the partitioning, this is the right answer
        # this is probably an exceedingly rare case
        # TODO: can we prove this never happens?
        last_bound_start = self._partition_boundary_hi[-2] + 1
        if last_bound_start <= len(self.xs) - 1:
            if self.range_sum(last_bound_start, len(self.xs) - 1) == self._max_partition_size:
                # print(self.xs)
                # print(f'{self._max_partition_size=}')
                # print(self.range_sum(self._partition_boundary_hi[-2], len(self.xs) - 1))
                # print(self._partition_boundary_lo)
                # print(self._partition_boundary_hi)
                # print(self._max_partition_jump_table)
                self._min_partition_size = self._max_partition_size
                # return
                raise RuntimeError('unexpected optimization happened')

        # sanity check the boundaries
        assert len(self._partition_boundary_lo) == self.k + 1
        assert len(self._partition_boundary_hi) == self.k + 1
        assert self._partition_boundary_lo[0] == -1
        assert self._partition_boundary_hi[0] == -1
        assert self._partition_boundary_lo[-1] == len(self.xs) - 1
        assert self._partition_boundary_hi[-1] == len(self.xs) - 1  # this must have reached the end
        assert all((hi >= lo) for hi, lo in zip(self._partition_boundary_hi, self._partition_boundary_lo))

    def __narrow_partition_boundary_lists(self):
        """
        runs the partition boundary check code in reverse (from right to left)
        which helps tighten the constraints for the second half of the set of partitions

        note that this runs from right to left
        also the min partition increases by one, and the max partition decreases by one
        unless we found the answer in which case they will both be made equal
        """
        pointer_min_right = len(self.xs) - 1
        pointer_max_right = len(self.xs) - 1
        for _k in range(self.k - 1, 0, -1):
            # note that min partition has increased by one, hence < instead of <=
            assert self.range_sum(self._min_partition_reverse_jump_table[pointer_min_right], pointer_min_right
                                  ) < self._min_partition_size
            assert self.range_sum(self._max_partition_reverse_jump_table[pointer_max_right], pointer_max_right
                                  ) <= self._max_partition_size

            # move the min pointer backwards based on the incremented min partition size
            # since the jump table was built for the original min size, we check the size of one more item
            new_pointer_min_right = self._min_partition_reverse_jump_table[pointer_min_right]
            if new_pointer_min_right > 0:
                if self.range_sum(new_pointer_min_right - 1, pointer_min_right) <= self._min_partition_size:
                    new_pointer_min_right -= 1
            self._partition_boundary_hi[_k] = min(self._partition_boundary_hi[_k], new_pointer_min_right)
            if new_pointer_min_right > 0:
                new_pointer_min_right -= 1
            pointer_min_right = new_pointer_min_right

            # optimization: we check max_size minus one so we can reduce the search space by one more
            new_pointer_max_right = self._max_partition_reverse_jump_table[pointer_max_right]
            if new_pointer_max_right > 0:
                if self.range_sum(new_pointer_max_right, pointer_max_right) == self._max_partition_size:
                    new_pointer_max_right += 1
            self._partition_boundary_lo[_k] = max(self._partition_boundary_lo[_k], new_pointer_max_right)
            if new_pointer_max_right > 0:
                new_pointer_max_right -= 1
            pointer_max_right = new_pointer_max_right

            # this bound can accidentally push past lo, so if it happens, then just don't let it
            # moving the bound back means the first painter is being allocated less than P
            self._partition_boundary_hi[_k] = max(self._partition_boundary_hi[_k], self._partition_boundary_lo[_k])

        assert all((hi >= lo) for hi, lo in zip(self._partition_boundary_hi, self._partition_boundary_lo))

        # early exit if the incremented pointer min right succeeded, since if it partitions this is the answer
        if self.range_sum(0, pointer_min_right) <= self._min_partition_size:
            self._max_partition_size = self._min_partition_size
            return

        # if we had used the max partition size, pointer_max_right must reach 0 by next step,
        # and we could assert that `self._max_partition_reverse_jump_table[pointer_max_right] == 0`
        # but because we aren't using the max partition size, it's possible for the condition to fail,
        # in which case we've found that the minimum partition size is the current max, and hence is also the answer
        # early exit if the incremented pointer max right failed, otherwise decrement it
        if self.range_sum(0, pointer_max_right) > self._max_partition_size - 1:
            self._min_partition_size = self._max_partition_size
            return

        self._min_partition_size += 1  # this is now the original min partition size plus 2
        self._max_partition_size -= 1

    def __post_init__(self):
        """
        precompute the cumulative sum, max, and len
        (total sum is the last elem of cumulative sum)
        overall O(N) runtime if k < N
        """
        # number of workers must be non-negative
        if self.k < 0:
            raise ValueError(f'found invalid value {self.k} for `k`, which must be non-negative (k >= 0)')
        # edge case
        if len(self.xs) == 0:
            return  # no work needs to be done

        # 1st O(N) pass: loop to precompute all the properties of xs
        self.__build_cumulative_sum()

        # early exit if the list was empty
        if not self.xs:
            return

        # early exit if no workers exist to do work
        if self.k == 0:
            raise ValueError(f'found invalid value {self.k} for `k`, which must be >0 when xs is not empty')

        self.__calculate_partition_size_bounds()
        if self._min_partition_size == self._max_partition_size:
            return  # this also handles the case where k=1

        # 2nd O(N) pass: precompute jump tables
        self.__build_jump_tables()
        if self._min_partition_size == self._max_partition_size:
            return

        # 3rd O(K) pass: build partition boundary lookup tables
        self.__build_partition_boundary_lists()
        if self._min_partition_size == self._max_partition_size:
            return

        # 4th O(N) pass: tighten partition bounds by looking in reverse
        self.__narrow_partition_boundary_lists()
        if self._min_partition_size == self._max_partition_size:
            return

        # TODO: linked list of which partitions need to be checked (i.e. which boundaries are ambiguous)
        # maybe use a dict of int->int as pointers, and it probably needs to be doubly linked
        # can probably safely ignore the GC and just not remove old nodes

    def range_sum(self, start: int, end: int) -> int:
        # O(1) lookup via cumulative sum
        # includes start and end
        assert start >= 0
        assert end >= start, (start, end, len(self.xs))
        assert len(self._cumulative_sum) >= end + 1, (start, end, len(self._cumulative_sum))

        _range_sum = self._cumulative_sum[end]
        if start > 0:
            _range_sum -= self._cumulative_sum[start - 1]
        return _range_sum

    def test_partition(self, partition_size: int) -> bool:
        """
        if too small (i.e., couldn't fit into the partitions), returns False
        if it fits (or is too big), returns True

        TODO: consider if we can return an "excess amount" or the extra space
        returning zero iff the last partition is completely full
        this is a sort of gradient that can inform the outer loop to make better guesses
        """
        start_idx = 0
        n = len(self.xs)

        # TODO: keep a record of all the discovered partition points
        # so we can update the hi and lo partition boundaries
        # if we exit early we can still update the ones we found so far
        # partition_boundaries = {}

        # TODO: use the linked list instead of iterating over all of k
        # when we skip a k we just take the prev k lo since that boundary is fixed
        # at the start we can assert the size is correct

        for _k in range(self.k):
            # note that the first elem of `partition_boundary` is the start of the 0-th partition, we need the end
            end_idx_lo = max(self._min_partition_jump_table[start_idx], self._partition_boundary_lo[_k + 1])
            end_idx_hi = min(self._max_partition_jump_table[start_idx], self._partition_boundary_hi[_k + 1])

            # We want to find the largest index 'p' in [lo_idx, hi_idx]
            # such that range_sum(current_idx, p) <= partition_size.
            # range_sum(current_idx, p) = cumsum[p] - (cumsum[current_idx-1])
            # So: cumsum[p] <= partition_size + (cumsum[current_idx-1])

            prev_sum = self.range_sum(0, start_idx - 1) if _k else 0
            target = prev_sum + partition_size

            # bisect_right returns insertion point.
            # We search in `_cumulative_sum` within the bounds [lo_idx, hi_idx + 1]
            # (Note: +1 because bisect range is lo, hi exclusive at end)
            next_idx = bisect.bisect_right(self._cumulative_sum,
                                           target,
                                           lo=end_idx_lo,
                                           hi=min(end_idx_hi + 1, n),
                                           ) - 1

            # so it turns out that the range might be constrained in a way the partition cannot happen
            # but bisect must still output something, even if it is incorrect
            # if this o(1) sanity check fails, we just exit false
            if self.range_sum(start_idx, next_idx) > partition_size:
                return False

            # If we reached the end of the array, we are done
            if next_idx >= n - 1:
                return True

            start_idx = next_idx + 1

            # If current_idx went past the bounds, something is wrong or finished (handled above)
            if start_idx >= n:
                return True

        # TODO: update partition hi or lo before returning
        # this should probably be its own class method - pass the dict and let it update

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

        # TODO: consider whether test partition should return a gradient of some sort
        # so we can determine a more intelligent next guess

        if not self.xs: return 0

        # note that min and max partition sizes are both inclusive
        # this is different from the standard bisect's assumptions/invariants
        while self._min_partition_size < self._max_partition_size:
            mid = (self._min_partition_size + self._max_partition_size) // 2

            # we assume nobody else is updating the min and max partition sizes
            # TODO: if we update from inside tst_partition, don't update here, or constrain via min/max
            if self.test_partition(mid):
                self._max_partition_size = mid  # to keep the range inclusive, we don't use max=mid-1
            else:
                self._min_partition_size = mid + 1
        return self._min_partition_size


def painter(xs, k):
    # O(len(xs) * log(sum(xs))
    if sum(xs) == 0: return 0

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


    def tst_painter(xs, k, attempt=0, trials=0):
        # renamed to avoid pytest detection
        if trials:
            print(f'[{attempt + 1}/{trials}]', len(xs), k)  # , xs)
        else:
            print(len(xs), k, xs)
        t = time.perf_counter()
        solver = PaintersPartitionSolver(xs=xs, k=k)
        print('precompute', time.perf_counter() - t)
        answer = solver.solve_partition()
        print('precompute + solver', time.perf_counter() - t)
        t = time.perf_counter()
        ans2 = painter(xs, k)
        print('dp', time.perf_counter() - t)
        print(answer)
        assert ans2 == answer, ans2
        print('-' * 100)


    for _i in range(10):
        for j in range(10):
            for test_k in range(1, 10):
                tst_painter([_i] * j, test_k)
    for _i in range(10):
        for j in range(10):
            for test_k in range(1, 10):
                tst_painter(list(range(_i)) * j, test_k)

    for _attempt in range(_trials := 100):
        test_xs = [random.randint(1, 1_000_000_000) for _ in range(random.randint(1, 1000_000))]
        test__k = random.randint(1, 1_000)
        tst_painter(test_xs, test__k, _attempt, _trials)
