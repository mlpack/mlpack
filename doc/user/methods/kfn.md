## `KFN`: k-furthest-neighbor search

The `KFN` class implements k-furthest neighbor search, a core computational task
that is useful in many machine learning situations.  Either exact or approximate
furthest neighbors can be computed.  mlpack's `KFN` class uses
[trees](../core/trees.md), by default the [`KDTree`](../core/trees/kdtree.md),
to provide significantly accelerated computation; depending on input options, an
efficient dual-tree or single-tree algorithm is used.

<!-- An image showing a simple reference set and query set. -->
<div style="text-align: center">
<svg width="500" height="250" viewBox="0 0 500 250" fill="none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <!-- Border. -->
  <line x1="0"   y1="0"   x2="500" y2="0"   stroke="black" />
  <line x1="500" y1="0"   x2="500" y2="250" stroke="black" />
  <line x1="500" y1="250" x2="0"   y2="250" stroke="black" />
  <line x1="0"   y1="250" x2="0"   y2="0"   stroke="black" />

  <!-- Lines between points. -->
  <line x1="360" y1="170" x2="15" y2="175" stroke="black" stroke-dasharray="2" />
  <line x1="110" y1="135" x2="425" y2="215"  stroke="black" stroke-dasharray="2" />

  <!-- Five reference points. -->
  <circle cx="100" cy="55" r="5" fill="#880000" />
  <circle cx="70"  cy="10" r="5" fill="#880000" />
  <circle cx="425" cy="215" r="5" fill="#880000" />
  <circle cx="15"  cy="175" r="5" fill="#880000" />
  <circle cx="200" cy="220" r="5" fill="#880000" />
  <text x="115" y="55"  text-anchor="middle" fill="black" font-style="italic">r₀</text>
  <text x="85"  y="10"  text-anchor="middle" fill="black" font-style="italic">r₁</text>
  <text x="440" y="220" text-anchor="middle" fill="black" font-style="italic">r₂</text>
  <text x="30"  y="190" text-anchor="middle" fill="black" font-style="italic">r₃</text>
  <text x="215" y="225" text-anchor="middle" fill="black" font-style="italic">r₄</text>

  <!-- Two query points. -->
  <circle cx="360" cy="170" r="5" fill="#000088" />
  <circle cx="110" cy="135" r="5" fill="#000088" />
  <text x="375" y="175" text-anchor="middle" fill="black" font-style="italic">q₀</text>
  <text x="125" y="130" text-anchor="middle" fill="black" font-style="italic">q₁</text>
</svg>
<p style="font-size: 85%">
The exact furthest neighbor of the query point <i>q₀</i> is <i>r₃</i>.
<br />
The exact furthest neighbor of the query point <i>q₁</i> is <i>r₂</i>.
<br />
Approximate search will not return exact neighbors, but results will be close;
<br />
e.g., <i>r₁</i> could be returned as the approximate furthest neighbor of
<i>q₀</i>.
</p>
</div>

Given a _reference set_ of points and a _query set_ of queries, the `KFN` class
will compute the furthest neighbors in the reference set of every point in the
query set.  If no query set is given, then `KFN` will find the furthest
neighbors of every point in the reference set; this is also called the
_all-furthest-neighbors_ problem.

<!-- TODO: link to ApproxKFN and DrusillaSelect as alternatives for higher dimensions -->

The `KFN` class supports configurable behavior, with numerous runtime and
compile-time parameters, including the distance metric, type of data, search
strategy, and tree type.

#### Simple usage example:

```c++
// Compute the 5 exact furthest neighbors of every point of random numeric data.

// All data is uniform random: 10-dimensional data.  Replace with a data::Load()
// call or similar for a real application.
arma::mat referenceSet(10, 1000, arma::fill::randu); // 1000 points.

mlpack::KFN kfn;                     // Step 1: create object.
kfn.Train(referenceSet);             // Step 2: set the reference set.
arma::mat distances;
arma::Mat<size_t> neighbors;
kfn.Search(5, neighbors, distances); // Step 3: find 5 furthest neighbors of
                                     //         every point in `referenceSet`.

// Note: you can also call `kfn.Search(querySet, 5, neighbors, distances)` to
// find the furthest neighbors in `referenceSet` of a different set of points.

// Print some information about the results.
std::cout << "Found " << neighbors.n_rows << " neighbors for each of "
    << neighbors.n_cols << " points in the dataset." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `KFN` objects.
 * [Search strategies](#search-strategies): details of search strategies
   supported by `KFN`.
 * [Setting the reference set (`Train()`)](#setting-the-reference-set-train):
   set the dataset that will be searched for furthest neighbors.
 * [Searching for neighbors](#searching-for-neighbors): call `Search()` to
   compute furthest neighbors (exact or approximate).
 * [Computing quality metrics](#computing-quality-metrics) to determine how
   accurate the computed furthest neighbors are, if approximate search was used.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-template-parameters) for
   configuring behavior, including distance metrics, tree types, and different
   element types.
 * [Advanced examples](#advanced-examples) that make use of custom template
   parameters.

#### See also:

 * [`KNN` (k-nearest-neighbors)](knn.md)
 * [mlpack trees](../core/trees.md)
 * [mlpack geometric algorithms](../modeling.md#geometric-algorithms)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

### Constructors

 * `kfn = KFN()`
 * `kfn = KFN(strategy=DUAL_TREE, epsilon=0)`
   - Construct a `KFN` object, optionally using the given `strategy` for search
     and `epsilon` for maximum relative approximation level.
   - This does not set the reference set to be searched!
     [`Train()`](#setting-the-reference-set-train) must be called before calling
     [`Search()`](#searching-for-neighbors).

 * `kfn = KFN(referenceSet)`
 * `kfn = KFN(referenceSet, strategy=DUAL_TREE, epsilon=0)`
   - Construct a `KFN` object on the given set of reference points, using the
     given `strategy` for search and `epsilon` for maximum relative
     approximation level.
   - This will build a [`KDTree`](../core/trees/kdtree.md) with default
     parameters on `referenceSet`, if `strategy` is not `NAIVE`.
   - If `referenceSet` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceSet)`) to avoid copying `referenceSet`.  The dataset
     will still be accessible via [`ReferenceSet()`](#other-functionality), but
     points may be in shuffled order.

 * `kfn = KFN(referenceTree)`
 * `kfn = KFN(referenceTree, strategy=DUAL_TREE, epsilon=0)`
   - Construct a `KFN` object with a pre-built tree `referenceTree`, which
     should be of type `KFN::Tree` (a convenience typedef of
     [`KDTree`](../core/trees/kdtree.md) that uses
     [`FurthestNeighborStat`](../core/trees/binary_space_tree.md#furthestneighborstat)
     as its
     [`StatisticType`](../core/trees/binary_space_tree.md#statistictype)).
   - The search strategy will be set to `strategy` and maximum relative
     approximation level will be set to `epsilon`.
   - If `referenceTree` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceTree)`) to avoid copying `referenceTree`.  The tree
     will still be accessible via [`ReferenceTree()`](#other-functionality).

***Note:*** if `std::move()` is not used to pass `referenceSet` or
`referenceTree`, those objects will be copied---which can be expensive!  Be sure
to use `std::move()` if possible.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `referenceSet` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix containing dataset to search for furthest neighbors in. | _(N/A)_ |
| `referenceTree` | `KFN::Tree` (a [`KDTree`](../core/trees/kdtree.md)) | Pre-built kd-tree on reference data. | _(N/A)_ |
| `strategy` | `enum NeighborSearchStrategy` | The search strategy that will be used when `Search()` is called.  Must be one of `NAIVE`, `SINGLE_TREE`, `DUAL_TREE`, or `GREEDY_SINGLE_TREE`.  [More details.](#search-strategies) | `DUAL_TREE` |
| `epsilon` | `double` | Allowed relative approximation error.  `0` means exact search.  Must be non-negative. | `0.0` |

***Notes:***

 - By default, exact furthest neighbors are found.  When `strategy` is
   `SINGLE_TREE` or `DUAL_TREE`, set `epsilon` to a positive value to enable
   approximation (higher `epsilon` means more approximation is allowed).  See
   more in [Search strategies](#search-strategies), below.

 - If constructing a tree manually, the `KFN::Tree` type can be used (e.g.,
   `tree = KFN::Tree(referenceData)`).  `KFN::Tree` is a convenience typedef of
   either [`KDTree`](../core/trees/kdtree.md) or the chosen `TreeType` if
   [custom template parameters](#advanced-functionality-template-parameters) are
   being used.

### Search strategies

The `KFN` class can search for furthest neighbors using one of the following
four strategies.  These can be specified in the constructor as the `strategy`
parameter, or by calling `kfn.SearchStrategy() = strategy`.

 * `DUAL_TREE` _(default)_: two trees will be used at search time with a
   [dual-tree algorithm (pdf)](https://ratml.org/pub/pdf/2013tree.pdf) to
   allow the maximum amount of pruning.
   - This is generally the fastest strategy for exact search in low-to-medium
     dimensions.
   - Backtracking search is performed to find either exact furthest neighbors,
     or approximate furthest neighbors if `kfn.Epsilon() > 0`.

 * `SINGLE_TREE`: a tree built on the reference points will be traversed once
   for each point whose furthest neighbors are being searched for.
   - Single-tree search generally empirically
     [scales logarithmically](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Space_partitioning).
   - Backtracking search is performed to find either exact furthest neighbors,
     or approximate furthest neighbors if `kfn.Epsilon() > 0`.

 * `GREEDY_SINGLE_TREE`: for each point whose furthest neighbors are being
   searched for, a tree built on the reference points will be traversed in a
   greedy manner---recursing directly and only to the furthest node in the tree
   to find furthest neighbor candidates.
   - The approximation level with this strategy cannot be controlled; the
     setting of `kfn.Epsilon()` is ignored.
   - Greedy single-tree search scales logarithmically (e.g. `O(log N)` for each
     point whose neighbors are being computed, if the size of the reference set
     is `N`); however, since no backtracking is performed, results are obtained
     *extremely* efficiently.
   - This strategy is most effective when
     [spill trees](../core/trees/sp_tree.md) are used; to do this, use
     `SPTree` or another [spill tree variant](../core/trees/spill_tree.md) as
     the
     [`TreeType` template parameter](#advanced-functionality-template-parameters).

 * `NAIVE`: brute-force search---for each point whose furthest neighbors are
   being searched for, compute the distance to *every* point in the reference
   set.
   - This strategy always gives exact results; the setting of `kfn.Epsilon()` is
     ignored.
   - Brute-force search scales poorly, with a runtime cost of `O(N)` per point,
     where `N` is the size of the reference set.
   - However, brute-force search does not suffer from
     [poor performance in high dimensions](https://en.wikipedia.org/wiki/K-d_tree#Degradation_in_performance_with_high-dimensional_data) as trees often do.
   - When this strategy is used, no tree structure is used.

### Setting the reference set (`Train()`)

If the reference set was not set in the constructor, or if it needs to be
changed to a new reference set, the `Train()` method can be used.

 * `kfn.Train(referenceSet)`
   - Set the reference set to `referenceSet`.
   - This will build a [`KDTree`](../core/trees/kdtree.md) with default
     parameters on `referenceSet`, if `strategy` is not
     [`NAIVE`](#search-strategies).
   - If `referenceSet` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceSet)`) to avoid copying `referenceSet`.  The dataset
     will still be accessible via [`ReferenceSet()`](#other-functionality), but
     points may be in shuffled order.

 * `kfn.Train(referenceTree)`
   - Set the reference tree to `referenceTree`, which should be of type
     `KFN::Tree` (a convenience typedef of [`KDTree`](../core/trees/kdtree.md)
     that uses
     [`FurthestNeighborStat`](../core/trees/binary_space_tree.md#furthestneighborstat)
     as its
     [`StatisticType`](../core/trees/binary_space_tree.md#statistictype)).
   - If `referenceTree` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceTree)`) to avoid copying `referenceTree`.  The tree
     will still be accessible via [`ReferenceTree()`](#other-functionality).

### Searching for neighbors

Once the reference set and parameters are set, searching for furthest neighbors
can be done with the `Search()` method.

 * `kfn.Search(k, neighbors, distances)`
   - Search for the `k` furthest neighbors of all points in the reference set
     (e.g. [`kfn.ReferenceSet()`](#other-functionality)), storing the results in
     `neighbors` and `distances`.
   - `neighbors` and `distances` will be set to have `k` rows and
     `kfn.ReferenceSet().n_cols` columns.
   - `neighbors(i, j)` (e.g. the `i`th row and `j`th column of `neighbors`) will
     hold the column index of the `i`th furthest neighbor of the `j`th point in
     `kfn.ReferenceSet()`.
   - That is, the `i`th furthest neighbor of `kfn.ReferenceSet().col(j)` is
     `kfn.ReferenceSet().col(neighbors(i, j))`.
   - `distances(i, j)` will hold the distance between the `j`th point in
     `kfn.ReferenceSet()` and its `i`th furthest neighbor.

 * `kfn.Search(querySet, k, neighbors, distances)`
   - Search for the `k` furthest neighbors in the reference set of all points in
     `querySet`, storing the results in `neighbors` and `distances`.
   - `neighbors` and `distances` will be set to have `k` rows and
     `querySet.n_cols` columns.
   - `neighbors(i, j)` (e.g. the `i`th row and `j`th column of `neighbors`) will
     hold the column index in `kfn.ReferenceSet()` of the `i`th furthest
     neighbor of the `j`th point in `querySet`.
   - That is, the `i`th furthest neighbor of `querySet.col(j)` is
     `kfn.ReferenceSet().col(neighbors(i, j))`.
   - `distances(i, j)` will hold the distance between the `j`th point in
     `querySet` and its `i`th furthest neighbor in `kfn.ReferenceSet()`.

***Notes***:

 * If `kfn.Epsilon() > 0` and the search strategy is
   [`DUAL_TREE` or `SINGLE_TREE`](#search-strategies), then
   the search will return approximate furthest neighbors within a relative
   distance of `kfn.Epsilon()` of the true furthest neighbor.

 * `kfn.Epsilon()` is ignored when the search strategy is
   [`GREEDY_SINGLE_TREE` or `NAIVE`](#search-strategies).

 * When using a `queryTree` multiple times, the bounds in the tree must be
   reset.  Call `kfn.ResetTree(queryTree)` after each call to `Search()` to
   reset the bounds, or call `node.Stat().Reset()` on each node in `queryTree`.

 * When searching for approximate neighbors, it is possible that no furthest
   neighbor candidate will be found.  If this is true, then the corresponding
   element in `neighbors` will be set to `SIZE_MAX` (e.g. `size_t(-1)`), and the
   corresponding element in `distances` will be set to `0`.

---

#### Search Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `querySet` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix of query points for which the furthest neighbors in the reference set should be found. |
| `k` | `size_t` | Number of furthest neighbors to search for. |
| `neighbors` | [`arma::Mat<size_t>`](../matrices.md) | Matrix to store indices of furthest neighbors into.  Will be set to size `k` x `N`, where `N` is the number of points in the query set (if specified), or the reference set (if not). |
| `distances` | [`arma::mat`](../matrices.md) | Matrix to store distances to furthest neighbors into.  Will be set to the same size as `neighbors`. |

### Computing quality metrics

If approximate furthest neighbor search is performed (e.g. if
`kfn.Epsilon() > 0`), and exact furthest neighbors are known, it is possible to
compute quality metrics of the approximate search.

 * `double error = kfn.EffectiveError(computedDistances, exactDistances)`
   - Given a matrix of exact distances and computed approximate distances, both
     with the same size (rows equal to `k`, columns equal to the number of
     points in the query or reference set), compute the average relative error
     of the computed distances.
   - Any neighbors with distance 0 (e.g. no furthest neighbor candidate found)
     in `exactDistances` will be ignored for the computation.
   - `computedDistances` and `exactDistances` should be matrices produced by
     [`kfn.Search()`](#searching-for-neighbors).
   - When dual-tree or single-tree search was used, `error` will be no greater
     than [`kfn.Epsilon()`](#other-functionality).

 * `double recall = kfn.Recall(computedNeighbors, trueNeighbors)`
   - Given a matrix containing indices of exact furthest neighbors and computed
     approximate neighbors, both with the same size (rows equal to `k`, columns
     equal to the number of points in the query or reference set), compute the
     recall (percentage of true neighbors found).
   - `computedNeighbors` and `trueNeighbors` should be matrices produced by
     [`kfn.Search()`](#searching-for-neighbors).
   - The recall will be between `0.0` and `1.0`, with `1.0` indicating perfect
     recall.

### Other functionality

 - `kfn.ReferenceSet()` will return a `const arma::mat&` representing the data
   points in the reference set.  This matrix cannot be modified.
   * If a
     [custom `MatType` template parameter](#advanced-functionality-template-parameters)
     has been specified, then the return type will be `const MatType&`.

 - `kfn.ReferenceTree()` will return a `KFN::Tree*` (a
   [`KDTree`](../core/trees/kdtree.md) with
   [`FurthestNeighborStat`](../core/trees/binary_space_tree.md#furthestneighborstat)
   as the
   [`StatisticType`](../core/trees/binary_space_tree.md#statistictype)).
   * This is the tree that will be used at search time, if the search strategy
     is not [`NAIVE`](#search-strategies).
   * If the search strategy was [`NAIVE`](#search-strategies) when the
     object was constructed, then `kfn.ReferenceTree()` will return `nullptr`.
   * If a
     [custom `TreeType` template parameter](#advanced-functionality-template-parameters)
     has been specified, then `KFN::Tree*` will be that type of tree, not a
     `KDTree`.

 - `kfn.SearchStrategy()` will return the [search strategy](#search-strategies)
   that will be used when `kfn.Search()` is called.
   * `kfn.SearchStrategy() = newStrategy` will set the
     [search strategy](#search-strategies) to `newStrategy`.
   * `newStrategy` must be one of the supported search strategies.

 - `kfn.Epsilon()` returns a `double` representing the allowed level of
   approximation.  If `0` and `kfn.SearchStrategy()` is either dual- or
   single-tree search, then `kfn.Search()` will return exact results.
   * `kfn.Epsilon() = eps` will set the allowed level of approximation to `eps`.
   * `eps` must be greater than or equal to `0.0`.

 - After calling `kfn.Search()`, `kfn.BaseCases()` will return a `size_t`
   representing the number of point-to-point distance computations that were
   performed, if a [tree-traversing search strategy](#search-strategies) was
   used.

 - After calling `kfn.Search()`, `kfn.Scores()` will return a `size_t`
   indicating the number of tree nodes that were visited during search, if a
   [tree-traversing search strategy](#search-strategies) was used.

 - A `KFN` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).  Note
   that for large reference sets, this will also serialize the dataset
   (`kfn.ReferenceSet()`) and the tree (`kfn.Tree()`), and so the resulting file
   may be quite large.

 - `KFN::Tree` is a convenience typedef representing the type of the tree that
   is used for searching.
   * By default, this is a [`KDTree`](../core/trees/kdtree.md); specifically:
     `KFN::Tree` is `KDTree<EuclideanDistance, FurthestNeighborStat, arma::mat>`.
   * If a
     [custom `TreeType`, `DistanceType`, and/or `MatType`](#advanced-functionality-template-parameters)
     are specified, then
     `KFNType<DistanceType, TreeType, MatType>::Tree = TreeType<DistanceType, FurthestNeighborStat, MatType>`.
   * A custom tree can be built and passed to
     [`Train()`](#setting-the-reference-set-train) or the
     [constructor](#constructors) with, e.g., `tree = KFN::Tree(referenceSet)`
     or `tree = KFN::Tree(std::move(referenceSet))`.

### Simple examples

Find the exact furthest neighbor of every point in the `cloud` dataset.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Construct the KFN object; this will avoid copies via std::move(), and build a
// kd-tree on the dataset.
mlpack::KFN kfn(std::move(dataset));

arma::mat distances;
arma::Mat<size_t> neighbors;

// Compute the exact furthest neighbor.
kfn.Search(1, neighbors, distances);

// Print the furthest neighbor and distance of the fifth point in the dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - Point values: " << kfn.ReferenceSet().col(4).t();
std::cout << " - Index of furthest neighbor: " << neighbors(0, 4) << "."
    << std::endl;
std::cout << " - Distance to furthest neighbor: " << distances(0, 4) << "."
    << std::endl;
```

---

Split the `corel-histogram` dataset into two sets, and find the exact furthest
neighbor in the first set of every point in the second set.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Split the dataset into two equal-sized sets randomly with `data::Split()`.
arma::mat referenceSet, querySet;
mlpack::data::Split(dataset, referenceSet, querySet, 0.5);

// Construct the KFN object, building a tree on the reference set.  Copies are
// avoided by the use of `std::move()`.
mlpack::KFN kfn(std::move(referenceSet));

arma::mat distances;
arma::Mat<size_t> neighbors;

// Compute the exact furthest neighbor in `referenceSet` of every point in
// `querySet`.
kfn.Search(querySet, 1, neighbors, distances);

// Print information about the dual-tree traversal.
std::cout << "Dual-tree traversal computed " << kfn.BaseCases()
    << " point-to-point distances and " << kfn.Scores()
    << " tree node-to-tree node distances." << std::endl;

// Print information about furthest neighbors of the points in the query set.
std::cout << "The furthest neighbor of query point 3 is reference point index "
    << neighbors(0, 3) << ", with distance " << distances(0, 3) << "."
    << std::endl;
std::cout << "The L2-norm of reference point " << neighbors(0, 3) << " is "
    << arma::norm(kfn.ReferenceSet().col(neighbors(0, 3)), 2) << "."
    << std::endl;

// Compute the average furthest neighbor distance for all points in the query
// set.
const double averageDist = arma::mean(arma::vectorise(distances));
std::cout << "Average distance between a query point and its furthest "
    << "neighbor: " << averageDist << "." << std::endl;
```

---

Perform approximate single-tree search to find 5 furthest neighbors of the first
point in a subset of the `LCDM` dataset.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::data::Load("lcdm_tiny.csv", dataset);

// Build a KFN object on the LCDM dataset, and pass with `std::move()` so that
// we can avoid copying the dataset.  Set the search strategy to single-tree.
mlpack::KFN kfn(std::move(dataset), mlpack::SINGLE_TREE);

// Now we will compute the 5 furthest neighbors of the first point in the
// dataset.
arma::mat distances;
arma::Mat<size_t> neighbors;
kfn.Search(kfn.ReferenceSet().col(0), 5, neighbors, distances);

std::cout << "The five furthest neighbors of the first point in the LCDM "
    << "dataset are:" << std::endl;
for (size_t k = 0; k < 5; ++k)
{
  std::cout << " - " << neighbors(k, 0) << " (with distance " << distances(k, 0)
      << ")." << std::endl;
}
```

---

Use greedy single-tree search to find 5 approximate furthest neighbors of every
point in the `cloud` dataset.  Then, compute the exact furthest neighbors, and
use these to find the average error and recall of the approximate search.

***Note***: greedy single-tree search is far more effective when using spill
trees---see the [advanced examples](#advanced-examples) for another example that
does exactly that.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Build a tree on the dataset and set the search strategy to the greedy single
// tree strategy.
mlpack::KFN kfn(std::move(dataset), mlpack::GREEDY_SINGLE_TREE);

// Compute the 5 approximate furthest neighbors of every point in the dataset.
arma::mat distances;
arma::Mat<size_t> neighbors;
kfn.Search(5, neighbors, distances);

std::cout << "Greedy approximate kFN search computed " << kfn.BaseCases()
    << " point-to-point distances and visited " << kfn.Scores()
    << " tree nodes in total." << std::endl;

// Now switch to exact computation and compute the true neighbors and distances.
arma::Mat<size_t> trueNeighbors;
arma::mat trueDistances;
kfn.SearchStrategy() = mlpack::DUAL_TREE;
kfn.Epsilon() = 0.0;
kfn.Search(5, trueNeighbors, trueDistances);

// Compute the recall and effective error.
const double recall = kfn.Recall(neighbors, trueNeighbors);
const double effectiveError = kfn.EffectiveError(distances, trueDistances);

std::cout << "Recall of greedy search: " << recall << "." << std::endl;
std::cout << "Effective error of greedy search: " << effectiveError << "."
    << std::endl;
```

---

Build a `KFN` object on the `cloud` dataset and save it to disk.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Build the reference tree.
mlpack::KFN kfn(std::move(dataset));

// Save the KFN object to disk with the name 'kfn'.
mlpack::data::Save("kfn.bin", "kfn", kfn);

std::cout << "Successfully saved KFN model to 'kfn.bin'." << std::endl;
```

---

Load a `KFN` object from disk, and inspect the
[`KDTree`](../core/trees/kdtree.md) that is held in the object.

```c++
// Load the KFN object with name 'kfn' from 'kfn.bin'.
mlpack::KFN kfn;
mlpack::data::Load("kfn.bin", "kfn", kfn);

// Inspect the KDTree held by the KFN object.
std::cout << "The KDTree in the KFN object in 'kfn.bin' holds "
    << kfn.ReferenceTree().NumDescendants() << " points." << std::endl;
std::cout << "The root of the tree has " << kfn.ReferenceTree().NumChildren()
    << " children." << std::endl;
if (kfn.ReferenceTree().NumChildren() == 2)
{
  std::cout << " - The left child holds "
      << kfn.ReferenceTree().Child(0).NumDescendants() << " points."
      << std::endl;
  std::cout << " - The right child holds "
      << kfn.ReferenceTree().Child(1).NumDescendants() << " points."
      << std::endl;
}
```

---

Compute the 5 approximate furthest neighbors of two subsets of the
`corel-histogram` dataset using a pre-built query tree.  Then, reuse the query
tree to compute the exact neighbors and compute the effective error.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Split the covertype dataset into two parts of equal size.
arma::mat referenceSet, querySet;
mlpack::data::Split(dataset, referenceSet, querySet, 0.5);

// Build the KFN object, passing the reference set with `std::move()` to avoid a
// copy.  We use the default dual-tree strategy for search and set the maximum
// allowed relative error to 0.1 (10%).
mlpack::KFN kfn(std::move(referenceSet), mlpack::DUAL_TREE, 0.1);

// Now build a tree on the query points.  This is a KDTree, and we manually
// specify a leaf size of 50 points.  Note that the KDTree rearranges the
// ordering of points in the query set.
mlpack::KFN::Tree queryTree(std::move(querySet));

// Compute the 5 approximate furthest neighbors of all points in the query set.
arma::mat distances;
arma::Mat<size_t> neighbors;
kfn.Search(queryTree, 5, neighbors, distances);

// Now compute the exact neighbors---but since we are using dual-tree search and
// an externally-constructed query tree, we must reset the bounds!
arma::mat trueDistances;
arma::Mat<size_t> trueNeighbors;
kfn.ResetTree(queryTree);
kfn.Epsilon() = 0;
kfn.Search(queryTree, 5, trueNeighbors, trueDistances);

// Compute the effective error.
const double effectiveError = kfn.EffectiveError(distances, trueDistances);

std::cout << "Effective error of approximate dual-tree search was "
    << effectiveError << " (limit via kfn.Epsilon() was 0.1)." << std::endl;
```

---

<!-- TODO: add some KFN examples to the examples repository! -->

### Advanced functionality: template parameters

The `KFN` class is a typedef of the configurable `KFNType` class, which has five
template parameters that can be used for custom behavior.  The full signature of
the class is:

```
KFNType<DistanceType,
        TreeType,
        MatType,
        DualTreeTraversalType,
        SingleTreeTraversalType>
```

 * `DistanceType`: specifies the [distance metric](../core/distances.md) to be
   used for finding furthest neighbors.
 * `TreeType`: specifies the type of [tree](../core/trees.md) to be used for
   indexing points for fast tree-based search.
 * `MatType`: specifies the type of matrix used for representation of data.
 * `DualTreeTraversalType`: specifies the
   [traversal](../../developer/trees.md#traversals) strategy that will be used
   when searching with the dual-tree strategy.
 * `SingleTreeTraversalType`: specifies the
   [traversal](../../developer/trees.md#traversals) strategy that will be used
   when searching with the dual-tree strategy.

When custom template parameters are specified:

 * The `referenceSet` and `querySet` parameters to
   [the constructor](#constructors),
   [`Train()`](#setting-the-reference-set-train), and
   [`Search()`](#searching-for-neighbors) must have type `MatType` instead of
   `arma::mat`.
 * The `distances` parameter to [`Search()`](#searching-for-neighbors) should
   have type `MatType`.
 * The convenience typedef `Tree` (e.g. `KFNType<DistanceType, TreeType, MatType, DualTreeTraversalType, SingleTreeTraversalType>::Tree`) will be equivalent to
   `TreeType<DistanceType, FurthestNeighborStat, MatType>`.
 * All tree parameters (`referenceTree` and `queryTree`) should have type
   `TreeType<DistanceType, FurthestNeighborStat, MatType>`.

---

#### `DistanceType`

 * Specifies the distance metric that will be used when searching for furthest
   neighbors.

 * The default distance type is
   [`EuclideanDistance`](../core/distances.md#lmetric).

 * Many [pre-implemented distance metrics](../core/distances.md) are available
   for use, such as [`ManhattanDistance`](../core/distances.md#lmetric) and
   [`ChebyshevDistance`](../core/distances.md#lmetric) and others.

 * [Custom distance metrics](../../developer/distances.md) are easy to
   implement, but *must* satisfy the triangle inequality to provide correct
   results when searching with trees (e.g. `kfn.SearchStrategy()` is not
   `NAIVE`).
   - ***NOTE:*** the cosine distance ***does not*** satisfy the triangle
     inequality.

<!-- TODO: link to FastMKS or some other solution for the cosine distance -->

---

#### `TreeType`

 * Specifies the tree type that will be built on the reference set (and
   possibly query set), if `kfn.SearchStrategy()` is not `NAIVE`.

 * The default tree type is [`KDTree`](../core/trees/kdtree.md).

 * Numerous [pre-implemented tree types](../core/trees.md) are available for
   use.

 * [Custom trees](../../developer/trees.md) are very difficult to implement, but
   it is possible if desired.
   - If you have implemented a fully-working `TreeType` yourself, please
     contribute it upstream if possible!

---

#### `MatType`

 * Specifies the type of matrix to use for representing data (the reference set
   and the query set).

 * The default `MatType` is `arma::mat` (dense 64-bit precision matrix).

 * Any matrix type implementing the Armadillo API will work; so, for instance,
   `arma::fmat` or `arma::sp_mat` can also be used.

---

#### `DualTreeTraversalType`

 * Specifies the [traversal strategy](../../developer/trees.md#traversals) to
   use when performing a dual-tree search to find furthest neighbors (e.g. when
   `kfn.SearchStrategy()` is `DUAL_TREE`).

 * By default, the [`TreeType`](#treetype)'s default dual-tree traversal (e.g.
   `TreeType<DistanceType, FurthestNeighborStat, MatType>::DualTreeTraversalType`)
   will be used.

 * In general, this parameter does not need to be specified, except when a
   custom type of traversal is desired.
   - For instance, the [`SpillTree`](../core/trees/spill_tree.md) class provides
     the
     [`DefeatistDualTreeTraversal`](../core/trees/spill_tree.md#tree-traversals)
     strategy, which is a specific greedy strategy for spill trees when
     performing approximate nearest neighbor search (however, it can also be
     used for approximate furthest neighbor search!).

---

#### `SingleTreeTraversalType`

 * Specifies the [traversal strategy](../../developer/trees.md#traversals) to
   use when performing a single-tree search to find furthest neighbors (e.g.
   when `kfn.SearchStrategy()` is `SINGLE_TREE`).

 * By default, the [`TreeType`](#treetype)'s default dual-tree traversal (e.g.
   `TreeType<DistanceType, FurthestNeighborStat, MatType>::SingleTreeTraversalType`)
   will be used.

 * In general, this parameter does not need to be specified, except when a
   custom type of traversal is desired.
   - For instance, the [`SpillTree`](../core/trees/spill_tree.md) class provides
     the
     [`DefeatistSingleTreeTraversal`](../core/trees/spill_tree.md#tree-traversals)
     strategy, which is a specific greedy strategy for spill trees when
     performing approximate nearest neighbor search (however, it can also be
     used for approximate furthest neighbor search!).

---

### Advanced examples

Perform exact furthest neighbor search to find the furthest neighbor of every
point in the `cloud` dataset, using 32-bit floats to represent the data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Construct the KFN object; this will avoid copies via std::move(), and build a
// kd-tree on the dataset.
mlpack::KFNType<mlpack::EuclideanDistance, mlpack::KDTree, arma::fmat> kfn(
    std::move(dataset));

arma::fmat distances; // This type is arma::fmat, just like the dataset.
arma::Mat<size_t> neighbors;

// Compute the exact furthest neighbor.
kfn.Search(1, neighbors, distances);

// Print the furthest neighbor and distance of the fifth point in the dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - Point values: " << kfn.ReferenceSet().col(4).t();
std::cout << " - Index of furthest neighbor: " << neighbors(0, 4) << "."
    << std::endl;
std::cout << " - Distance to furthest neighbor: " << distances(0, 4) << "."
    << std::endl;
```

---

Perform approximate single-tree furthest neighbor search using the Chebyshev
(L-infinity) distance as the distance metric.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Construct the KFN object; this will avoid copies via std::move(), and build a
// kd-tree on the dataset.
mlpack::KFNType<mlpack::ChebyshevDistance> kfn(std::move(dataset),
    mlpack::SINGLE_TREE, 0.2);

arma::mat distances;
arma::Mat<size_t> neighbors;

// Compute the exact furthest neighbor.
kfn.Search(1, neighbors, distances);

// Print the furthest neighbor and distance of the fifth point in the dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - Point values: " << kfn.ReferenceSet().col(4).t();
std::cout << " - Index of approximate furthest neighbor: " << neighbors(0, 4)
    << "." << std::endl;
std::cout << " - Chebyshev distance to approximate furthest neighbor: "
    << distances(0, 4) << "." << std::endl;
```

---

Use an [Octree](../core/trees/octree.md) (a tree known to be faster in very few
dimensions) to find the exact furthest neighbors of all the points in a tiny
subset of the 3-dimensional LCDM dataset.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::data::Load("lcdm_tiny.csv", dataset);

// Construct the KFN object with Octrees.
mlpack::KFNType<mlpack::EuclideanDistance, mlpack::Octree> kfn(
    std::move(dataset));

arma::mat distances;
arma::Mat<size_t> neighbors;

// Find the exact furthest neighbor of every point.
kfn.Search(1, neighbors, distances);

// Print the average, minimum, and maximum furthest neighbor distances.
std::cout << "Average furthest neighbor distance: " <<
    arma::mean(arma::vectorise(distances)) << "." << std::endl;
std::cout << "Minimum furthest neighbor distance: " <<
    arma::min(arma::vectorise(distances)) << "." << std::endl;
std::cout << "Maximum furthest neighbor distance: " <<
    arma::max(arma::vectorise(distances)) << "." << std::endl;
```

---

Using a 32-bit floating point representation, split the `lcdm_tiny` dataset into
a query and a reference set, and then use `KFN` with preconstructed random
projection trees ([`RPTree`](../core/trees/rp_tree.md)) to find the 5
approximate furthest neighbors of each point in the query set with the Manhattan
distance.  Then, compute exact furthest neighbors and the average error and
recall.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::fmat dataset;
mlpack::data::Load("lcdm_tiny.csv", dataset);

// Split the dataset into a query set and a reference set (each with the same
// size).
arma::fmat referenceSet, querySet;
mlpack::data::Split(dataset, referenceSet, querySet, 0.5);

// This is the type of tree we will build on the datasets.
using TreeType = mlpack::RPTree<mlpack::ManhattanDistance,
                                mlpack::FurthestNeighborStat,
                                arma::fmat>;

// Note that we could also define TreeType as below (it is the same type!):
//
// using TreeType = mlpack::KFNType<mlpack::ManhattanDistance,
//                                  mlpack::RPTree,
//                                  arma::fmat>::Tree;

// We build the trees here with std::move() in order to avoid copying data.
//
// For RPTrees, this reorders the points in the dataset, but if original indices
// are needed, trees can be constructed with mapping objects.  (See the RPTree
// documentation for more details.)
TreeType referenceTree(std::move(referenceSet));
TreeType queryTree(std::move(querySet));

// Construct the KFN object with the prebuilt reference tree.
mlpack::KFNType<mlpack::ManhattanDistance, mlpack::RPTree, arma::fmat> kfn(
    std::move(referenceTree), mlpack::DUAL_TREE, 0.1 /* max 10% error */);

// Find 5 approximate furthest neighbors.
arma::fmat approxDistances;
arma::Mat<size_t> approxNeighbors;
kfn.Search(queryTree, 5, approxNeighbors, approxDistances);
std::cout << "Computed approximate neighbors." << std::endl;

// Now compute exact neighbors.  When reusing the query tree, this requires
// resetting the statistics inside the query tree manually.
arma::fmat exactDistances;
arma::Mat<size_t> exactNeighbors;
kfn.ResetTree(queryTree);
kfn.Epsilon() = 0.0; // Error tolerance is now 0% (exact search).
kfn.Search(queryTree, 5, exactNeighbors, exactDistances);
std::cout << "Computed exact neighbors." << std::endl;

// Compute error measures.
const double recall = kfn.Recall(approxNeighbors, exactNeighbors);
const double effectiveError = kfn.EffectiveError(approxDistances,
    exactDistances);

std::cout << "Recall of approximate search: " << recall << "." << std::endl;
std::cout << "Effective relative error of approximate search: "
    << effectiveError << " (vs. limit of 0.1)." << std::endl;
```

---

Use [spill trees](../core/trees/sp_tree.md) to perform greedy single-tree
approximate furthest neighbor search on the `cloud` dataset, and compare with
the other spill tree traversers and exact results.  Compare with the results in
the [simple examples](#simple-examples) section where the default
[`KDTree`](../core/trees/kdtree.md) is used---spill trees perform significantly
better for greedy search!

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Build a spill tree on the dataset and set the search strategy to the greedy
// single tree strategy.  We will build the tree manually, so that we can
// configure the build-time parameters (see the SPTree documentation for more
// details).
using TreeType = mlpack::SPTree<mlpack::EuclideanDistance,
                                mlpack::FurthestNeighborStat,
                                arma::mat>;
TreeType referenceTree(std::move(dataset), 10.0 /* tau, overlap parameter */);

mlpack::KFNType<mlpack::EuclideanDistance,
                mlpack::SPTree,
                arma::mat,
                // Use the special defeatist spill tree traversers.
                TreeType::template DefeatistDualTreeTraverser,
                TreeType::template DefeatistSingleTreeTraverser> kfn(
    std::move(referenceTree), mlpack::GREEDY_SINGLE_TREE);

arma::mat greedyDistances, dualDistances, singleDistances, exactDistances;
arma::Mat<size_t> greedyNeighbors, dualNeighbors, singleNeighbors,
    exactNeighbors;

// Compute the 5 approximate furthest neighbors of every point in the dataset.
kfn.Search(5, greedyNeighbors, greedyDistances);

std::cout << "Greedy approximate kFN search computed " << kfn.BaseCases()
    << " point-to-point distances and visited " << kfn.Scores()
    << " tree nodes in total." << std::endl;

// Now do the same thing, but with defeatist dual-tree search.  Note that
// defeatist dual-tree search is not backtracking, so we don't need to set
// kfn.Epsilon().
kfn.SearchStrategy() = mlpack::DUAL_TREE;
kfn.Search(5, dualNeighbors, dualDistances);

std::cout << "Dual-tree approximate kFN search computed " << kfn.BaseCases()
    << " point-to-point distances and visited " << kfn.Scores()
    << " tree nodes in total." << std::endl;

// Finally, use defeatist single-tree search.
kfn.SearchStrategy() = mlpack::SINGLE_TREE;
kfn.Search(5, singleNeighbors, singleDistances);

std::cout << "Single-tree approximate kFN search computed " << kfn.BaseCases()
    << " point-to-point distances and visited " << kfn.Scores()
    << " tree nodes in total." << std::endl;

// Now switch to the exact naive strategy and compute the true neighbors and
// distances.
kfn.SearchStrategy() = mlpack::NAIVE;
kfn.Epsilon() = 0.0;
kfn.Search(5, exactNeighbors, exactDistances);

// Compute the recall and effective error for each strategy.
const double greedyRecall = kfn.Recall(greedyNeighbors, exactNeighbors);
const double dualRecall = kfn.Recall(dualNeighbors, exactNeighbors);
const double singleRecall = kfn.Recall(singleNeighbors, exactNeighbors);

const double greedyError = kfn.EffectiveError(greedyDistances, exactDistances);
const double dualError = kfn.EffectiveError(dualDistances, exactDistances);
const double singleError = kfn.EffectiveError(singleDistances, exactDistances);

// Print the results.  To tune the results, try constructing the SPTrees
// manually and specifying different construction parameters.
std::cout << std::endl;
std::cout << "Recall with spill trees:" << std::endl;
std::cout << " - Greedy search:      " << greedyRecall << "." << std::endl;
std::cout << " - Dual-tree search:   " << dualRecall << "." << std::endl;
std::cout << " - Single-tree search: " << singleRecall << "." << std::endl;
std::cout << std::endl;
std::cout << "Effective error with spill trees:" << std::endl;
std::cout << " - Greedy search:      " << greedyError << "." << std::endl;
std::cout << " - Dual-tree search:   " << dualError << "." << std::endl;
std::cout << " - Single-tree search: " << singleError << "." << std::endl;
```
