# linear time algorithm for the painter's partition problem

## how it works

1. pre-processing (linear time)
   * cumulative sum
   * find max
   * build lookup table for range of `min_partition := max(math.ceil(sum(xs) / k), max(xs))`  
   * build lookup table for range of `max_partition := min_partition + math.ceil(sum(xs) / k)`
   * optional: pre-build `lo,hi` ranges by partitioning using `min_partition` and `min_partition + 1` from both ends
2. 