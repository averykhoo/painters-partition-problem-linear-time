# linear time algorithm for the painter's partition problem

## how it works

1. error handling
    * `if any(x <= 0 for x in xs): raise ValueError`
        * technically, `x==0` is solvable: filter these values out, keeping track of their original locations (in a
          pre-pre-processing step, since it might still be a trivial case), then restore them in a post-processing step
    * `if k <= 0: raise ValueError`
   2. trivial/edge cases (simple `O(n)` solutions)
       * `if len(xs) == 0: return 0` (painters are allowed to do no work - alternatively just add a dummy 0 to xs)
       * `if 1 <= len(xs) <= k: return max(xs)` (alternatively pad with dummy 0 items)
       * `if k == 1: return sum(xs)`
       * `if k == 2:` fairly trivial `O(n)` "meet-in-the middle" algorithm, just start from both ends
       * `if k == 3:` first calculate `sum(xs)` then reuse the `k==2` code but with a middle segment
       * `if k == len(xs) + 1:` merge the two smallest neighboring elements, then return the new `max(xs)`
           * with recursion and clever indexing, this would be at best a `O(len(xs) * log(len(xs)))` general solution
   3. pre-processing (`O(n)`)
       * cumulative sum (and total sum, but that's just the last element)
         * also note that a range sum is just one elem minus the other
       * find max
       * calculate `min_partition` and `max_partition`
           * optional optimization: do a quick test to see whether `min_partition` works
           * only requires a couple extra vars in the loop
   4. binary search within binary search (`O(k * log(sum(xs)) * log(len(xs)))`)
       * `pointer = 0`
       * binary search with `lo = min_partition` and `hi = max_partition`
           * `for partition_idx in range(k):`
               * binary search using `lo = min_partition_lookup[pointer]` and `hi = max_partition_lookup[pointer]`
               * `if new_pointer >= len(xs):` depends if we're at the last partition
   5. optional optimizations for lower amortized time
       * pre-build `lo,hi` ranges by partitioning using `min_partition + 1` from the opposite end
           * and `min_partition` from the normal end
           * if this reaches the end then we can exit early
       * update ranges on-the-fly at each outer binary search run
       * exit the inner loop early if we hit any `hi`, since the partition automatically succeeds
           * or falls below `lo`, since the partition fails
       * if `k <= len(xs)` then just return `max(xs)`
       * if we try a partition size `a` and the largest partition is `b`, but it's still too big,
         then exclude from `b`, not from `a`
       * keep a (linked) list of which partitions are still defined as ranges, removing any that become strictly defined,
         which helps reduce the strictly O(K * log(whatever)) per iteration to something potentially less than K
           * but i'm not sure how to math the runtime


we can also partition using max partition from both sides (maybe max minus one from the other side) and then take the strictest bounds we find over the 4 runs - this probably won't be as helpful but I'm not sure how to quantify the uselessness of it

## why it works

### invariants that always hold

> * `max(xs) >= sum(xs)/len(xs)` (otherwise the solution is trivial)
> * `sum(xs) > len(xs)` (otherwise the solution is trivial)
> * `len(xs) >= k` (otherwise the solution is trivial)
> * `P >= sum(xs)/k`
> * `P >= max(xs)`
> * `P <= sum(xs)`

### invariant when `P == max(xs)`

maybe the concept should be called "inefficient packing" since we are packing the list xs into k containers of equal size, but we're adversarially designing xs to pack as poorly as possible

constraints are that:
* this packing must use xs in continuous order, and k is now the minimum number of containers needed to hold xs
* the length of xs and k are assumed to be finite but are variables, and we use the max and sum (and maybe mean?) to come up with invariants that must hold

one set of invariants is where the ideal partition size P is equal to the max element:
above we derived a formula that relates the rest of xs to a largest k, no matter what the distribution or count of elements are

consider how *empty* the partitions can possibly be if every painter is allocated at least 1 for some optimal partition size P and number of workers k
for some optimal partitioning P, there must be at least one slot full, i.e. there exists at least one painter allocated P area
if we allocate the rest P/2+1 but put the max last and allocate that last painter 1, then the total work done is `P+1+(k-2)(P/2+1)` = `0.5(kP + 2k - 2)` which turns out not to be emptiest allocation 
if the rest are allocated P and 1 alternately, the total work done is (for an odd number k) `((k-1)/2)(P+1)+1 = 0.5(kP + k + 1 - P)`
or for even k `(k/2)(P+1) = 0.5(kP + k)`
this defines the largest k needed to hold a total partition size - no matter the length of xs, if we sub in max(xs) and k, if we know that sum(xs) is <= to the sum then we know the answer is just max(xs)

now knowing that `P = max(xs)` we can find the relation that `sum(xs)<=0.5(k*max(xs)+k)` if k is even or `sum(xs)<=0.5(k*max(xs)+k+1-max(xs))` if k is odd - if either of these hold then the answer is just max(xs)

hence we can return immediately when `max(xs) >= (2*sum(xs)/k) - 1` (if k is even) or `max(xs) >= (2*(sum(xs)-1)/(k-1)) - 1` (if k is odd)
we could be lazy and knowing that `max(xs)>k` we can just not check evenness and use the even check
but also it's a tighter bound to truncate the first or last element along with one painter and call it an even length
we can safely drop one item and know we have a non-zero number because if k==1 or len(xs)==1 the answer was alr trivial

> if `max(xs) >= (2*sum(xs)/k) - 1`
> or if (`k%2==1` and `max(xs) >= (2*(sum(xs)-max(xs[0], xs[-1]))/(k-1)) - 1`)
> or if `k >= len(xs) >= 1`
> then return `P=max(xs)`

### invariant when `P > max(xs)`

okay another invariant, this time `min(xs) < max(xs)` < `sum(xs)/k` <= `P`, i.e. on average each painter does more than `max(xs)`
this means that the most space we can "waste" is `(max(xs)-1) * (k-1)`, since we assume at least k containers are needed and at least one is full
also this means that the contents of each container are all filled with elements of equal size `max(xs)` except one that has some extra items of total size max(xs)-1
in this case we can have a multiple of items of size `max(xs)` in each bucket, but the full bucket has one additional item `max(xs)-1`
if we subtract that item, then all buckets are now optimally packed, so we know that means `P - (max(xs)-1) = (sum(xs)-(max(xs)-1))/k`
so when `sum(xs)/k` > `max(xs)` then we know that `P<=sum(xs)/k + (max(xs)-1) * (k-1)/k`

the upper bound probably got simplified to find `P` <= `max(xs)+sum(xs)/k` since removing the other terms keeps the invariant 
this simplification also helps it hold true when `min(xs) == max(xs)`

alternative derivation: `P*k` = total allocated paint = used paint + waste = `sum(xs)` + `(max(xs)-1)*(k-1)`

> if `max(xs) < sum(xs)/k)`
> then `P<=ceil(sum(xs)/k + (max(xs)-1) * (k-1)/k)`

### worst case conditions (todo, POSSIBLY WRONG):

* `max(xs) ~= sum(xs) / k`
* `max(xs) / mean(xs) ~= len(xs) / k`

[//]: # (TODO: can't remember why the last term here is sum<xs>/k <-- used <> since brackets cant be used in a comment)

* overall i think the complexity is `O(len(xs)) + O(k * log(len(xs)/k) * log(sum(xs)/k))`
  * can't figure out how to remove the dependency on the sum of items in xs
  * e.g. if all the items are random uint64s, then even if the list has length 10, the `log(mean(xs))` term dominates
  * at least it's certain that this is less than `O(sum(xs))`
  * i suppose it's kind of fair to say this is linear in terms of the length of the list in binary
  * and *technically* even reading the list from memory does that this much complexity so we're not much worse off

## fancy math

* attempt 3:
    * maybe there's a clever way to skip impossible numbers when dealing with large paintings
    * but simple preprocessing seems to be O(nlogn) so that's moot
* attempt 2:
    * `O(len(xs)) + O(k * log(len(xs)/k) * log(max(xs)))`
    * if we assume a uniform distribution, then `max(xs) ~= mean(xs) * 2`
    * and running the first attempt in reverse then `O(len(xs)) + O(k * log(len(xs)/k) ** 2)` which is `O(len(xs))`
    * but this requires a uniform distribution
* attempt 1:
    * `O(len(xs)) + O(k * log(len(xs)/k) * (log(len(xs)/k) + log(mean(xs))))`
    * `O(len(xs)) + O(k * log(len(xs)/k) ** 2) + O(k * log(len(xs)/k) * log(mean(xs)))`
    * and since `O(k * log(len(xs)/k) ** 2) << O(len(xs))` we can drop that term
    * but is `O(k * log(len(xs)/k) * log(mean(xs)))` less than `O(len(xs))`?
    * `O(k * log(max(xs) / mean(xs)) * log(mean(xs)))` is maximized when `mean(xs) ** 2 == max(xs)`
    * so this simplifies to `O(k * log(max(xs) / mean(xs)) ** 2) == O(k * log(len(xs)/k) ** 2)`
* proof for `max_partition <= 2 * math.ceil((sum(xs) - min(xs[0], xs[-1])) / (k - 1))`
    * efficient packing invariants:
        * the total sum of every pair of partitions must be at least partition + 1
        * the total size of all partitions is hence at most (roughly) 2 * sum(xs)
* proof for the `math.ceil(sum(xs) / k) + max(xs) - 1` case
    * basically guarantees a packing of at least mean(xs) into each partition
* proof that `O(k * log(sum(xs)/k) * log(len(xs)/k)) < O(sum(xs))`
    * and hopefully also `< O(len(xs))` but i'll need to remove the `mean(xs)` (or `max(xs)`) factor somehow
    * which might be safe since if we use the same datatype for `x` and `len(xs)` then they're both kinda bounded?
        * but it feels like a cop-out to assume that `log(max(xs)) ~= constant` because of the data type
* notation:
    * `xs` is a list of paintings, from `x₀` to `xₙ₋₁` (or `xs[0]` to `xs[-1] == xs[n-1]` for ascii compat)
        * `n` is `len(xs)`, and `N` is `sum(xs)`
    * `k` is the number of partitions (or painters)
    * `p` is partition size, `p̂` is optimal partition size (maybe `P` for better ascii compat?)
    * `log` is base 2
    *

# TODO

* make the math more precise
  * `max_partition` is also upper bounded as the shortest contiguous length of boards `len(xs) - k + 1`
  * something about the pigeonhole principle should help prove better runtime when `k` approaches `len(xs)`,
    since the possible range for each partition has to shrink proportionally to `len(xs) - k`
    * like we can prove that runtime is strictly no worse than `(len(xs)-1)C(k-1) = (len(xs)-1)!/((k-1)!(len(xs)-k))!)`
    * which should be further bounded by the range of values in `xs`