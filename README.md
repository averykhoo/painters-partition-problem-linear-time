# linear time algorithm for the painter's partition problem

## how it works

1. pre-processing (linear time)
    * cumulative sum (and total sum, but that's just the last element)
    * find max
    * build lookup table for range of `min_partition := max(math.ceil(sum(xs) / k), max(xs))`
    * build lookup table for range of `max_partition := min_partition + math.ceil(sum(xs) / k)`
2. binary search within binary search
    * todo
3. optional optimizations for lower amortized time
    * pre-build `lo,hi` ranges by partitioning using `min_partition` and `min_partition + 1` from both ends
    * update ranges on-the-fly at each outer binary search run

## why it works

* as long as `max(xs) < math.ceil(sum(xs) / k)`, the partition size will always fall between `math.ceil(sum(xs) / k)`
  and `2 * math.ceil(sum(xs) / k)`
* if `max(xs) >= math.ceil(sum(xs) / k)`, then the partition size will always fall between `max(xs)`
  and `max(xs) + math.ceil(sum(xs) / k)`
    * the upper bound can be optimized further, but doesn't matter in the proof of `O log(n)` runtime