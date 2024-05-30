# linear time algorithm for the painter's partition problem

## how it works

1. trivial cases
    * `if any(x <= 0 for x in xs): raise ValueError`
    * `if k < 1: raise ValueError`
    * `if len(xs) == 0: return 0`
    * `if k == 1: return sum(xs)`
    * `if len(xs) <= k: return max(xs)`
2. pre-processing (linear time)
    * cumulative sum (and total sum, but that's just the last element)
    * find max
    * build lookup table for range of `min_partition := max(math.ceil(sum(xs) / k), max(xs))`
    * build lookup table for range of `max_partition := max(2 * math.ceil((sum(xs) - xs[-1]) / (k - 1)), max(xs))`
3. binary search within binary search
    * todo
4. optional optimizations for lower amortized time
    * pre-build `lo,hi` ranges by partitioning using `min_partition` and `min_partition + 1` from both ends
        * if this reaches the end then we can exit early
    * update ranges on-the-fly at each outer binary search run
    * exit the inner loop early if we hit any `hi`, since the partition automatically succeeds
        * or falls below `lo`, since the partition fails
    * if `k <= len(xs)` then just return `max(xs)`

## why it works

* as long as (the math isn't exact here, TODO) `max(xs) < 2 * math.ceil(sum(xs) / k)`,
  then the partition size will always fall between `math.ceil(sum(xs) / k)` and `2 * math.ceil(sum(xs) / k)`
* if `max(xs) >= math.ceil(sum(xs) / k)`, then the partition size will always fall between `max(xs)`
  and `max(xs) + math.ceil(sum(xs) / k)`
    * the upper bound can be optimized further, but doesn't matter in the proof of `O log(n)` runtime
* so there's a gap between the upper and lower bound of about `sum(xs) / k` which is lower-bounded by `len(xs) / k`,
  and upper-bounded by `max(xs) * len(xs) / k`
* doing a binary search over this twice multiplied by `k` is (kind of) `O(k * log(N/k) ** 2)`
    * TODO: figure out how to remove `max(xs)` from the factor
    * maybe if we assume that `log(len(xs)/k) * log(max(xs)/k) ~= len(xs) * constant?`
    * which might be safe since if we use the same datatype for `x` and `len(xs)` then they're both kinda bounded?
* and since `O(log(n)) ** 2 < O(n)`, we have `O(k * N/k)` which is basically `O(N)`
* also there's preprocessing which is `O(N)`
* where `N == len(xs)`