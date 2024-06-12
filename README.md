# linear time algorithm for the painter's partition problem

## how it works

1. error handling
    * `if any(x <= 0 for x in xs): raise ValueError`
        * technically, `x==0` is solvable: filter these values out, keeping track of their original locations (in a
          pre-pre-processing step, since it might still be a trivial case), then restore them in a post-processing step
    * `if k <= 0: raise ValueError`
2. trivial/edge cases (simple `O(n)` solutions)
    * `if len(xs) == 0: return 0`
    * `if len(xs) <= k: return max(xs)`
    * `if k == 1: return sum(xs)`
    * `if k == 2:` fairly trivial `O(n)` "meet-in-the middle" algorithm, just start from both ends
    * `if k == 3:` first calculate `sum(xs)` then reuse the `k==2` code but with a middle segment
    * `if k == len(xs) + 1:` merge the two smallest neighboring elements, then return the new `max(xs)`
        * with recursion and some clever indexing, this should be an `O(len(xs) * log(len(xs)))` general solution
3. pre-processing (`O(n)`)
    * cumulative sum (and total sum, but that's just the last element)
    * find max
    * build lookup tables for ranges of `min_partition` and `max_partition`
4. binary search within binary search (`O(k * log(sum(xs)) * log(len(xs)))`)
    * `pointer = 0`
    * binary search with `lo = min_partition` and `hi = max_partition`
        * `for partition_idx in range(k):`
            * binary search using `lo = min_partition_lookup[pointer]` and `hi = max_partition_lookup[pointer]`
            * `if new_pointer >= len(xs):` depends if we're at the last partition
5. optional optimizations for lower amortized time
    * pre-build `lo,hi` ranges by partitioning using `min_partition` and `min_partition + 1` from both ends
        * if this reaches the end then we can exit early
    * update ranges on-the-fly at each outer binary search run
    * exit the inner loop early if we hit any `hi`, since the partition automatically succeeds
        * or falls below `lo`, since the partition fails
    * if `k <= len(xs)` then just return `max(xs)`

## why it works

* if `max(xs) < math.ceil(sum(xs) / k)`,
  then `min_partition = math.ceil(sum(xs) / k)`
  and `max_partition = math.ceil(sum(xs) / k) + max(xs)`
* if `math.ceil(sum(xs) / k) < max(xs) < 2 * math.ceil((sum(xs) - 1) / (k - 1))`,
  then `min_partition = max(xs)`
  and `max_partition = 2 * math.ceil((sum(xs) - 1) / (k - 1))`
* if `max(xs) >= 2 * math.ceil((sum(xs) - 1) / (k - 1))`,
  then `min_partition = max_partition = math.ceil(sum(xs) / k)`

* slightly more exact boundary conditions:
    * (`min_partition`) `math.ceil(sum(xs) / k)`
    * (`min_partition`) `max(xs)`
    * (`max_partition`) `math.ceil(sum(xs) / k) + max(xs) - 1`
    * (`max_partition` if `len(xs)` is even) `2 * math.ceil(sum(xs) / k) - 1`
    * (`max_partition` if `len(xs)` is odd) `2 * math.ceil((sum(xs) - max(xs[0], xs[-1])) / (k - 1)) - 1`
    * (`max_partition` ignoring length) `2 * math.ceil((sum(xs) - min(xs[0], xs[-1])) / (k - 1)) - 1`
    * (`max_partition`) `max(xs)`

* worst case conditions (todo):
    * `max(xs) ~= sum(xs) / k`
    * `max(xs) / mean(xs) ~= len(xs) / k`
    *

* overall i think the complexity is `O(len(xs)) + O(k * log(len(xs)/k) * log(sum(xs)/k))`
    * can't figure out how to remove the dependency on the sum of items in xs
    * e.g. if all the items are random uint64s, then even if the list has length 10, the `log(mean(xs))` term dominates
    * at least it's certain that this is less than `O(sum(xs))`
    * attempt 1:
        * `O(len(xs)) + O(k * log(len(xs)/k) * (log(len(xs)/k) + log(mean(xs))))`
        * `O(len(xs)) + O(k * log(len(xs)/k) ** 2) + O(k * log(len(xs)/k) * log(mean(xs)))`
        * and since `O(k * log(len(xs)/k) ** 2) << O(len(xs))` we can drop that term
        * but is `O(k * log(len(xs)/k) * log(mean(xs)))` less than `O(len(xs))`?
        * `O(k * log(max(xs) / mean(xs)) * log(mean(xs)))` is maximized when `mean(xs) -> 2`
        * maybe taking it as a constant factor of 2 is acceptable?
    * attempt 2:
        * `O(len(xs)) + O(k * log(len(xs)/k) * log(max(xs)))`
        * if we assume a uniform distribution, then `max(xs) ~= mean(xs) * 2`
        * and running the first attempt in reverse then `O(len(xs)) + O(k * log(len(xs)/k) ** 2)` which is `O(len(xs))`
        * but this requires a uniform distribution
    * attempt 3:
        * maybe there's a clever way to skip impossible numbers when dealing with large paintings
        * but simple preprocessing seems to be O(nlogn) so that's moot 

## fancy math

* proof for `max_partition <= 2 * math.ceil((sum(xs) - min(xs[0], xs[-1])) / (k - 1))`
    * efficient packing invariants:
        * the total sum of every pair of partitions must be at least partition + 1
        * the total size of all partitions is hence at most (roughly) 2 * sum(xs)
* proof for the `math.ceil(sum(xs) / k) + max(xs) - 1` case
    * basically guarantees a packing of at least mean(xs) into each partition
* proof that `O(k * log(sum(xs)) * log(len(xs))) < O(sum(xs))`
    * and hopefully also `< O(len(xs))` but i'll need to remove the `mean(xs)` (or `max(xs)`) factor somehow
    * which might be safe since if we use the same datatype for `x` and `len(xs)` then they're both kinda bounded?
        * but it feels like a cop-out to assume that `log(max(xs)) ~= constant` because of the data type

# TODO

* need to standardize the notation - is it `N = sum(xs)` or `N = len(xs)`? how about `k`?
* make the math more precise