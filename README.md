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

* as long as `max(xs) < math.ceil(sum(xs) / k)`, the partition size will always fall between `math.ceil(sum(xs) / k)`
  and `2 * math.ceil(sum(xs) / k)`
* if `max(xs) >= math.ceil(sum(xs) / k)`, then the partition size will always fall between `max(xs)`
  and `max(xs) + math.ceil(sum(xs) / k)`
    * the upper bound can be optimized further, but doesn't matter in the proof of `O log(n)` runtime