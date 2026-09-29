## `RangeSearch`

The `RangeSearch` class implements range search, a distance-based task that
finds all points in a set that are within a certain distance of a query point.
mlpack's `RangeSearch` class uses [trees](../core/trees.md), by default the
[`KDTree`](../core/trees/kdtree.md), to provide significantly accelerated
computation; depending on input options, an efficient dual-tree or single-tree
algorithm is used.

<!-- An image showing range search on a simple reference and query set. -->
<div style="text-align: center">
<svg width="500" height="250" viewBox="0 0 500 250" fill="none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
  <!-- Border. -->
  <line x1="0"   y1="0"   x2="500" y2="0"   stroke="black" />
  <line x1="500" y1="0"   x2="500" y2="250" stroke="black" />
  <line x1="500" y1="250" x2="0"   y2="250" stroke="black" />
  <line x1="0"   y1="250" x2="0"   y2="0"   stroke="black" />

  <!-- Circles indicating range to search for. -->
  <line x1="360" y1="170" x2="455" y2="170" stroke="black" stroke-dasharray="2"
/>
  <line x1="110" y1="135" x2="205" y2="135" stroke="black" stroke-dasharray="2"
/>
  <line x1="360" y1="170" x2="416.29165" y2="137.5" stroke="black" stroke-dasharray="2" />
  <line x1="110" y1="135" x2="166.29165" y2="102.5" stroke="black" stroke-dasharray="2" />
  <circle cx="360" cy="170" r="80" stroke="#00008833" stroke-width="30" fill="none" />
  <circle cx="110" cy="135" r="80" stroke="#00008833" stroke-width="30" fill="none" />
  <text x="390" y="145" text-anchor="middle" fill="black" font-style="italic">l</text>
  <text x="140" y="110" text-anchor="middle" fill="black" font-style="italic">l</text>
  <text x="400" y="185" text-anchor="middle" fill="black" font-style="italic">u</text>
  <text x="150" y="150" text-anchor="middle" fill="black" font-style="italic">u</text>

  <!-- Five reference points. -->
  <circle cx="100" cy="55" r="5" fill="#880000" />
  <circle cx="70"  cy="10" r="5" fill="#880000" />
  <circle cx="425" cy="215" r="5" fill="#880000" />
  <circle cx="35"  cy="175" r="5" fill="#880000" />
  <circle cx="200" cy="220" r="5" fill="#880000" />
  <text x="115" y="55"  text-anchor="middle" fill="black" font-style="italic">r₀</text>
  <text x="85"  y="10"  text-anchor="middle" fill="black" font-style="italic">r₁</text>
  <text x="440" y="220" text-anchor="middle" fill="black" font-style="italic">r₂</text>
  <text x="50"  y="180" text-anchor="middle" fill="black" font-style="italic">r₃</text>
  <text x="215" y="225" text-anchor="middle" fill="black" font-style="italic">r₄</text>

  <!-- Two query points. -->
  <circle cx="360" cy="170" r="5" fill="#000088" />
  <circle cx="110" cy="135" r="5" fill="#000088" />
  <text x="372" y="183" text-anchor="middle" fill="black" font-style="italic">q₀</text>
  <text x="122" y="148" text-anchor="middle" fill="black" font-style="italic">q₁</text>
</svg>
<p style="font-size: 85%">
Query point <i>q₀</i> only has reference point <i>r₂</i> in the range <i>[l,
u]</i>.
<br />
Query point <i>q₁</i> has reference points <i>r₀</i> and <i>r₃</i> in the range
<i>[l, u]</i>.
</p>
</div>

Given a _reference set_ of points, a _query set_ of queries, and a range
`[l, h]`, the `RangeSearch` class will, for each point in the query set, find
all the points in the reference set that have a distance `l <= d <= h` to that
query point.

The `RangeSearch` class supports configurable behavior, with numerous runtime
and compile-time parameters, including the distance metric, type of data, search
strategy, and tree type.

#### Simple usage example:

```c++
// Compute points in the distance range [0.75, 0.85] on random numeric data.

// All data is uniform random in 10 dimensions.  Replace with a Load() call or
// similar for a real application.
arma::mat referenceSet(10, 1000, arma::fill::randu); // 1000 points.

mlpack::RangeSearch rs;                 // Step 1: create object.
rs.Train(referenceSet);                 // Step 2: set the reference set.
std::vector<std::vector<size_t>> neighbors;
std::vector<std::vector<double>> distances;
rs.Search(mlpack::Range(0.75, 0.85),    // Step 3: find all points in the range
          neighbors, distances);        //         [0.75, 0.85].

// Note: you can also call `rs.Search(querySet, range, neighbors, distances)` to
// search with a separate set of query points.

// Print some information about the results.
size_t totalSize = 0;
for (size_t i = 0; i < neighbors.size(); ++i)
  totalSize += neighbors[i].size();
std::cout << "On average, each point has "
    << (double(totalSize) / neighbors.size()) << " points in the distance "
    << "range [0.75, 0.85]." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `RangeSearch` objects.
 * [Search strategies](#search-strategies): details of search strategies
   supported by `RangeSearch`.
 * [Setting the reference set (`Train()`)](#setting-the-reference-set-train):
   set the dataset that will be searched.
 * [Range searching](#range-searching): call `Search()` to find neighbors within
   a given distance range.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-template-parameters) for
   configuring behavior, including distance metrices, tree types, and different
   element types.
 * [Advanced examples](#advanced-examples) that make use of custom template
   parameters.

#### See also:

 * [mlpack trees](../core/trees.md)
 * [mlpack geometric algorithms](../modeling.md#geometric-algorithms)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)
 * [`KNN` (k-nearest-neighbors)](knn.md)

### Constructors

 * `rs = RangeSearch()`
 * `rs = RangeSearch(strategy=DUAL_TREE)`
   - Construct a `RangeSearch` object, optionally using the given `strategy` for
     search.
   - This does not set the reference set to be searched!
     [`Train()`](#setting-the-reference-set-train) must be called before
     [`Search()`](#range-searching).
    <!-- TODO: how to document `distance`? -->

 * `rs = RangeSearch(referenceSet)`
 * `rs = RangeSearch(referenceSet, strategy=DUAL_TREE)`
   - Construct a `RangeSearch` object on the given set of reference points,
     using the given `strategy` for search.
   - This will build a [`KDTree`](../core/trees/kdtree.md) with default
     parameters on `referenceSet`, if `strategy` is not `NAIVE`.
   - If `referenceSet` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceSet)`) to avoid copying `referenceSet`.  The dataset
     will still be accessible via [`ReferenceSet()`](#other-functionality), but
     points may be in a shuffled order.

 * `rs = RangeSearch(referenceTree)`
 * `rs = RangeSearch(referenceTree, strategy=DUAL_TREE)`
   - Construct a `RangeSearch` object with a pre-built tree `referenceTree`,
     which should be of type `RangeSearch::Tree` (a convenience typedef of
     [`KDTree`](../core/trees/kdtree.md)).
   - The search strategy will be set to `strategy`; for this constructor,
     `strategy` cannot be specified as `NAIVE` or an exception will be thrown.
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
| `referenceSet` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix containing dataset to search for nearest neighbors in. | _(N/A)_ |
| `referenceTree` | `RangeSearch::Tree` (a [`KDTree`](../core/trees/kdtree.md)) | Pre-built kd-tree on reference data. | _(N/A)_ |
| `strategy` | `enum TreeSearchStrategy` | The search strategy that will be used when `Search()` is called.  Must be one of `NAIVE`, `SINGLE_TREE`, or `DUAL_TREE`.  [More details.](#search-strategies) | `DUAL_TREE` |

***Notes:***

 - If constructing a tree manually, the `RangeSearch::Tree` type can be used
   (e.g. `tree = RangeSearch::Tree(referenceData)`).  `RangeSearch::Tree` is a
   convenience typedef of either [`KDTree`](../core/trees/kdtree.md) or the
   chosen `TreeType` if
   [custom template parameters](#advanced-functionality-template-parameters) are
   being used.

### Search strategies

The `RangeSearch` class can search for neighbors using one of the following
three strategies.  These can be specified in the constructor as the `strategy`
parameter, or by calling `rs.SearchStrategy() = strategy`.

 * `DUAL_TREE` _(default)_: two trees will be used at search time with a
   [dual-tree algorithm (pdf)](https://ratml.org/pub/pdf/2013tree.pdf) to
   allow the maximum amount of pruning.
   - This is generally the fastest strategy as it is able to prune candidate
     points for multiple query points simultaneously.
   - Backtracking search is performed to find neighbors within the specified
     distance range.

 * `SINGLE_TREE`: a tree built on the reference points will be traversed once
   for each query point.
   - Single-tree search generally empirically
     [scales logarithmically](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Space_partitioning).
   - Backtracking search is performed to find either exact nearest neighbors, or
     approximate nearest neighbors if `knn.Epsilon() > 0`.

 * `NAIVE`: brute-force search---for each query point, compute the distance to
   *every* point in the reference set.
   - Brute-force search scales poorly, with a runtime cost of `O(N)` per point,
     where `N` is the size of the reference set.
   - However, brute-force search does not suffer from
     [poor performance in high dimensions](https://en.wikipedia.org/wiki/K-d_tree#Degradation_in_performance_with_high-dimensional_data) as trees often do.
   - When this strategy is used, no tree structure is used.

### Setting the reference set (`Train()`)

If the reference set was not set in the constructor, or if it needs to be
changed to a new reference set, the `Train()` method can be used.

 * `rs.Train(referenceSet)`
   - Set the reference set to `referenceSet`.
   - This will build a [`KDTree`](../core/trees/kdtree.md) with default
     parameters on `referenceSet`, if `strategy` is not
     [`NAIVE`](#search-strategies).
   - If `referenceSet` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceSet)`) to avoid copying `referenceSet`.  The dataset
     will still be accessible via [`ReferenceSet()`](#other-functionality), but
     points may be in shuffled order.

 * `rs.Train(referenceTree)`
   - Set the reference tree to `referenceTree`, which should be of type
     `RangeSearch::Tree` (a convenience typedef of
     [`KDTree`](../core/trees/kdtree.md)).
   - If `referenceTree` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceTree)`) to avoid copying `referenceTree`.  The tree
     will still be accessible via [`ReferenceTree()`](#other-functionality).

### Range searching

Once the reference set and parameters are set, searching for neighbors within a
distance range can be done with the `Search()` method.

 * `rs.Search(range, neighbors, distances)`
   - Given `range` (a [`Range`](../core/math.md#range)), search for neighbors
     within the distance range `[range.Lo(), range.Hi()]` of each point in
     the reference set (e.g. [`rs.ReferenceSet()`](#other-functionality)),
     storing the results in `neighbors` and `distances`.
   - `neighbors`, a `std::vector<std::vector<size_t>>`, will be set to size
     `rs.ReferenceSet().n_cols` (e.g. one `std::vector<size_t>` for each
     reference point), and `neighbors[i]` will contain the list of neighbor
     indices within the distance range (in no particular order).
   - `distances`, a `std::vector<std::vector<double>>`, will be set to size
     `rs.ReferenceSet().n_cols` (e.g. one `std::vector<double>` for each
     reference point), and `distances[i]` will contain the distances between
     reference point `i` and each neighbor.
   - `neighbors[i][j]` will hold the column index of the `j`th neighbor of the
     `i`th point in `rs.ReferenceSet()`.
   - That is, the `j`th neighbor of `rs.ReferenceSet().col(i)` is
     `rs.ReferenceSet().col(neighbors[i][j])`.

 * `rs.Search(querySet, range, neighbors, distances)`
   - Given `range` (a [`Range`](../core/math.md#range)), search for neighbors
     within the distance range `[range.Lo(), range.Hi()]` of each point in
     `querySet`, storing the results in `neighbors` and `distances`.
   - `neighbors`, a `std::vector<std::vector<size_t>>`, will be set to size
     `querySet.n_cols` (e.g. one `std::vector<size_t>` for each query point),
     and `neighbors[i]` will contain the list of neighbor indices within the
     distance range (in no particular order).
   - `distances`, a `std::vector<std::vector<double>>`, will be set to size
     `querySet.n_cols` (e.g. one `std::vector<double>` for each query point),
     and `distances[i]` will contain the distances between query point `i` and
     each neighbor.
   - `neighbors[i][j]` will hold the column index of the `j`th neighbor of the
     `i`th point in `querySet`.
   - That is, the `j`th neighbor of `querySet.col(i)` is
     `rs.ReferenceSet().col(neighbors[i][j])`.

 * `rs.Search(queryTree, range, neighbors, distances, sameSet=false)`
   - Given `range` (a [`Range`](../core/math.md#range)), search for neighbors
     `queryTree`, search for neighbors within the distance range
     `[range.Lo(), range.Hi()]` of each point in `queryTree.Dataset()`, storing
     the results in `neighbors` and `distances`.
   - `neighbors`, a `std::vector<std::vector<size_t>>`, will be set to size
     `querySet.n_cols` (e.g. one `std::vector<size_t>` for each query point),
     and `neighbors[i]` will contain the list of neighbor indices within the
     distance range (in no particular order).
   - `distances`, a `std::vector<std::vector<double>>`, will be set to size
     `querySet.n_cols` (e.g. one `std::vector<double>` for each query point),
     and `distances[i]` will contain the distances between query point `i` and
     each neighbor.
   - `neighbors[i][j]` will hold the column index of the `j`th neighbor of the
     `i`th point in `queryTree.Dataset()`.
   - That is, the `j`th neighbor of `queryTree.Dataset().col(i)` is
     `rs.ReferenceSet().col(neighbors[i][j])`.
   - If `sameSet` is `true`, then the query set is understood to be the same as
     the reference set, and query points will not return their own index as part
     of the results.

***Notes***:

 * When `querySet` and `queryTree` are not specified, or when `sameSet` is
   `true`, a point will not return itself in the results, even if
   `range.Lo() == 0`.  However, if there are duplicate points `x` and `y` in the
   dataset, `y` will be returned as a neighbor of `x`, if `range.Lo() == 0`.

 * If `range` is too wide and the dataset is large, the `neighbors` and
   `distances` vectors may become huge, and search (even with trees) may take an
   infeasibly long time.  If `Search()` seems to be taking forever, try
   specifying a much smaller range, or try with one query point and see how
   large the results are, then adjust the range accordingly.

---

#### Search Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `querySet` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix of query points for which the neighbors in the reference set should be found. |
| `queryTree` | `RangeSearch::Tree` | Pre-built tree on query points to use for dual-tree search. |
| `range` | [`Range`](../core/math.md#range) | Distance range to search for. |
| `neighbors` | `std::vector<std::vector<size_t>>` | Vector of vectors to store neighbors into.  Will be set to length `N`, where `N` is the number of points in the query set (if specified), or the reference set (if not).  Each inner `std::vector<size_t>` will store the indices of neighbors of each query point. |
| `distances` | `std::vector<std::vector<double>>` | Vector of vectors to store distances into.  Will be set to the same size as `neighbors`.  Each inner `std::vector<double>` will store the distances between each query point and its found neighbors. |
| `sameSet` | `bool` | *(Only for `Search()` with a query set.)* If `true`, then `querySet` is the same set as the reference set. |

### Other functionality

 - `rs.ReferenceSet()` will return a `const arma::mat&` representing the data
   points in the reference set.  This matrix cannot be modified.
   * If a
     [custom `MatType` template parameter](#advanced-functionality-template-parameters)
     has been specified, then the return type will be `const MatType&`.

 - `rs.ReferenceTree()` will return a `RangeSearch::Tree*` (a
   [`KDTree`](../core/trees/kdtree.md)).
   * This is the tree that will be used at search time, if the search strategy
     is not [`NAIVE`](#search-strategies).
   * If the search strategy was [`NAIVE`](#search-strategies) when the object
     was constructed, then `rs.ReferenceTree()` will return `nullptr`.
   * If a
     [custom `TreeType` template parameter](#advanced-functionality-template-parameters)
     has been specified, then `RangeSearch::Tree` will be that type of tree,
     not a `KDTree`.

 - `rs.SearchStrategy()` will return the [search strategy](#search-strategies)
   that will be used when `rs.Search()` is called.
   * `rs.SearchStrategy() = newStrategy` will set the
     [search strategy](#search-strategies) to `newStrategy`.
   * `newStrategy` must be one of the supported search strategies.

 - After calling `rs.Search()`, `rs.BaseCases()` will return a `size_t`
   representing the number of point-to-point distance computations that were
   performed, if a [tree-traversing search strategy](#search-strategies) was
   used.

 - After calling `rs.Search()`, `rs.Scores()` will return a `size_t` indicating
   the number of tree nodes that were visited during search, if a
   [tree-traversing search strategy](#search-strategies) was used.

 - A `RangeSearch` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).  Note
   that for large reference sets, this will also serialize the dataset
   (`rs.ReferenceSet()`) and the tree (`rs.ReferenceTree()`), and so the
   resulting file may be quite large.

 - `RangeSearch::Tree` is a convenience typedef representing the type of the
   tree that is used for searching.
   * By default, this is a [`KDTree`](../core/trees/kdtree.md); specifically,
     `RangeSearch::Tree` is
     `KDTree<EuclideanDistance, EmptyStatistic, arma::mat>`.
   * If a
     [custom `TreeType`, `DistanceType`, and/or `MatType`](#advanced-functionality-template-parameters)
     are specified, then
     `RangeSearchType<DistanceType, MatType, TreeType>::Tree = TreeType<DistanceType, EmptyStatistic, MatType>`.
   * A custom tree can be built and passed to
     [`Train()`](#setting-the-reference-set-train) or the
     [constructor](#constructors) with, e.g.,
     `tree = RangeSearch::Tree(referenceSet)` or
     `tree = RangeSearch::Tree(std::move(referenceSet))`.

### Simple examples

Find all points with distances in the range `[10.0, 50.0]` of every other point
in the `cloud` dataset.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::Load("cloud.csv", dataset);

// Construct the RangeSearch object; this will avoid copies via std::move(), and
// build a kd-tree on the dataset.
mlpack::RangeSearch rs(std::move(dataset));

std::vector<std::vector<double>> distances;
std::vector<std::vector<size_t>> neighbors;

// Compute neighbors within the distance range [10.0, 50.0].
rs.Search(mlpack::Range(10.0, 50.0), neighbors, distances);

// Print information about the neighbors found for the fifth point in the
// dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - " << neighbors[4].size() << " neighbors in distance range "
    << "[10.0, 50.0]." << std::endl;
if (neighbors[4].size() > 0)
{
  std::cout << " - First neighbor: index " << neighbors[4][0] << ", distance "
      << distances[4][0] << "." << std::endl;
}
```

---

Split the `corel-histogram` dataset into two sets, and perform range search
using the first set as the reference set and the second as the query set.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv
arma::mat dataset;
mlpack::Load("corel-histogram.csv", dataset);

// Split the dataset into two equal-sized sets randomly with `Split()`.
arma::mat referenceSet, querySet;
mlpack::Split(dataset, referenceSet, querySet, 0.5);

// Construct the KNN object, building a tree on the reference set.  Copies are
// avoided by the use of `std::move()`.
mlpack::RangeSearch rs(std::move(referenceSet));

std::vector<std::vector<double>> distances;
std::vector<std::vector<size_t>> neighbors;

// Find all neighbors with distance between 1.20 and 1.25.
rs.Search(querySet, mlpack::Range(1.20, 1.25), neighbors, distances);

// Compute the maximum number of neighbors that any query point has, and the
// average.
size_t maxNeighbors = 0;
size_t sumNeighbors = 0;
for (size_t i = 0; i < neighbors.size(); ++i)
{
  maxNeighbors = std::max(maxNeighbors, neighbors[i].size());
  sumNeighbors += neighbors[i].size();
}
std::cout << "The maximum number of points in range [1.20, 1.25] that any query"
    << " point has is " << maxNeighbors << "." << std::endl;
std::cout << "The average number of neighbors in range [1.20, 1.25] that a "
    << " query point has is " << double(sumNeighbors) / neighbors.size() << "."
    << std::endl;
```

---

Use single-tree search to find points with distance less than 25.0 from the
first point of the `LCDM` dataset.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::Load("lcdm_tiny.csv", dataset);

// Build a RangeSearch object on the LCDM dataset, and pass with `std::move()`
// so that we can avoid copying the dataset.  The search strategy is set to
// single-tree search.
mlpack::RangeSearch rs(std::move(dataset), mlpack::SINGLE_TREE);

// Now compute points within distance 25.0 or less.
//
// NOTE: because the first point is in the reference set, and because we are
// passing a separate query set, RangeSearch will return point 0 as one of the
// neighbors.  This is an important caveat to be aware of when calling Search()
// with a query set.
std::vector<std::vector<double>> distances;
std::vector<std::vector<size_t>> neighbors;
rs.Search(rs.ReferenceSet().col(0), mlpack::Range(0.0, 25.0), neighbors,
    distances);

std::cout << "The first point in the LCDM dataset has " << neighbors[0].size()
    << " points within distance 25.0." << std::endl;
std::cout << "First five neighbors:" << std::endl;
for (size_t i = 0; i < std::min((size_t) 5, neighbors[0].size()); ++i)
{
  std::cout << " - Point index " << neighbors[0][i] << ", distance "
      << distances[0][i] << "." << std::endl;
}
```

---

Use brute-force search to find points with distance greater than 2500.0 on the
`cloud` dataset.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::Load("cloud.csv", dataset);

// Construct the RangeSearch object; this will avoid copies via std::move(), and
// will not build a tree, since we are specifying NAIVE as the search mode.
mlpack::RangeSearch rs(std::move(dataset), mlpack::NAIVE);

// Now search for any points with distance greater than 2500.0.
std::vector<std::vector<double>> distances;
std::vector<std::vector<size_t>> neighbors;
rs.Search(mlpack::Range(2500.0, DBL_MAX), neighbors, distances);

// Compute the maximum number of neighbors that any query point has, and the
// average.
size_t maxNeighbors = 0;
size_t sumNeighbors = 0;
for (size_t i = 0; i < neighbors.size(); ++i)
{
  maxNeighbors = std::max(maxNeighbors, neighbors[i].size());
  sumNeighbors += neighbors[i].size();
}
std::cout << "The maximum number of points with distance greater than 2500.0 "
    << "that any query point has is " << maxNeighbors << "." << std::endl;
std::cout << "The average number of neighbors with distance greater than "
    << "2500.0 that a query  point has is "
    << double(sumNeighbors) / neighbors.size() << "." << std::endl;
```

---

Build a `RangeSearch` object on the `cloud` dataset and save it to disk.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::Load("cloud.csv", dataset);

// Construct the RangeSearch object; this will avoid copies via std::move(), and
// will build a tree on the dataset.
mlpack::RangeSearch rs(std::move(dataset), mlpack::DUAL_TREE);

// Save the object to disk.
mlpack::Save("range_search.bin", rs);

std::cout << "Successfully saved RangeSearch model to 'range_search.bin'."
    << std::endl;
```

---

Load a `RangeSearch` object from disk, and inspect the
[`KDTree`](../core/trees/kdtree.md) that is held in the object.

```c++
// Load the RangeSearch object from 'range_search.bin'.
mlpack::RangeSearch rs;
mlpack::Load("range_search.bin", rs);

// Inspect the KDTree held by the RangeSearch object.
std::cout << "The KDTree in the RangeSearch object in 'range_search.bin' holds "
    << rs.ReferenceTree()->NumDescendants() << " points." << std::endl;
std::cout << "The root of the tree has " << rs.ReferenceTree()->NumChildren()
    << " children." << std::endl;
if (rs.ReferenceTree()->NumChildren() == 2)
{
  std::cout << " - The left child holds "
      << rs.ReferenceTree()->Child(0).NumDescendants() << " points."
      << std::endl;
  std::cout << " - The right child holds "
      << rs.ReferenceTree()->Child(1).NumDescendants() << " points."
      << std::endl;
}
```

### Advanced functionality: template parameters

The `RangeSearch` class is a templated class with tree template parameters that
can be used for custom behavior.  The full signature of the class is:

```
RangeSearch<DistanceType,
            MatType,
            TreeType>
```

 * `DistanceType`: specifies the [distance metric](../core/distances.md) to be
   used for finding nearest neighbors.
 * `MatType`: specifies the type of the matrix used for representation of data.
 * `TreeType`: specifies the type of [tree](../core/trees.md) to be used for
   indexing points for fast tree-based search.

When custom template parameters are specified:

 * The `referenceSet` and `querySet` parameters to
   [the constructor](#constructors),
   [`Train()`](#setting-the-reference-set-train), and
   [`Search()`](#range-searching) must have type `MatType` instead of
   `arma::mat`.
 * The `distances` parameter to [`Search()`](#range-searching) should
   have type `std::vector<std::vector<ElemType>>`, where `ElemType` is the
   element type held by the given `MatType` (e.g. `double` for `arma::mat`,
   `float` for `arma::fmat`, etc.).
 * The convenience typedef `Tree` (e.g.
   `RangeSearch<DistanceType, MatType, TreeType>::Tree`) will be equivalent to
   `TreeType<DistanceType, EmptyStatistic, MatType>`.
 * All tree parameters (`referenceTree` and `queryTree`) should have type
   `TreeType<DistanceType, EmptyStatistic, MatType>`.

---

#### `DistanceType`

 * Specifies the distance metric that will be used when range searching.

 * The default distance type is
   [`EuclideanDistance`](../core/distances.md#lmetric).

 * Many [pre-implemented distance metrics](../core/distances.md) are available
   for use, such as [`ManhattanDistance`](../core/distances.md#lmetric) and
   [`ChebyshevDistance`](../core/distances.md#lmetric) and others.

 * [Custom distance metrics](../../developer/distances.md) are easy to
   implement, but *must* satisfy the triangle inequality to provide correct
   results when searching with trees (e.g. `knn.SearchStrategy()` is not
   `NAIVE`).
   - ***NOTE:*** the cosine distance ***does not*** satisfy the triangle
     inequality.

---

#### `MatType`

 * Specifies the type of matrix to use for representing data (the reference set
   and the query set).

 * The default `MatType` is `arma::mat` (dense 64-bit precision matrix).

 * Any matrix type implementing the Armadillo API will work; so, for instance,
   `arma::fmat` or `arma::sp_mat` can also be used.

---

#### `TreeType`

 * Specifies the tree type that will be built on the reference set (and
   possibly query set), if `knn.SearchStrategy()` is not `NAIVE`.

 * The default tree type is [`KDTree`](../core/trees/kdtree.md).

 * Numerous [pre-implemented tree types](../core/trees.md) are available for
   use.

 * [Custom trees](../../developer/trees.md) are very difficult to implement, but
   it is possible if desired.
   - If you have implemented a fully-working `TreeType` yourself, please
     contribute it upstream if possible!

---

### Advanced examples

Find all points within the range `[10.0, 50.0]` of every point in the `cloud`
dataset, using 32-bit floats to represent the data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::Load("cloud.csv", dataset);

// Construct the RangeSearch object using arma::fmat as the MatType.
mlpack::RangeSearch<mlpack::EuclideanDistance, arma::fmat> rs(
    std::move(dataset));

// Note that when we use arma::fmat, the distance element type returned is now
// 'float' instead of 'double'.
std::vector<std::vector<float>> distances;
std::vector<std::vector<size_t>> neighbors;

// Compute neighbors within the distance range [10.0, 50.0].
rs.Search(mlpack::RangeType<float>(10.0, 50.0), neighbors, distances);

// Print information about the neighbors found for the fifth point in the
// dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - " << neighbors[4].size() << " neighbors in distance range "
    << "[10.0, 50.0]." << std::endl;
if (neighbors[4].size() > 0)
{
  std::cout << " - First neighbor: index " << neighbors[4][0] << ", distance "
      << distances[4][0] << "." << std::endl;
}
```

---

Perform range search on the `cloud` dataset using the Chebyshev (L-infinity)
distance as the distance metric.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::Load("cloud.csv", dataset);

// Construct the RangeSearch object using ChebyshevDistance as the DistanceType.
mlpack::RangeSearch<mlpack::ChebyshevDistance> rs(std::move(dataset));

std::vector<std::vector<double>> distances;
std::vector<std::vector<size_t>> neighbors;

// Compute neighbors within the distance range [10.0, 50.0].
rs.Search(mlpack::Range(10.0, 50.0), neighbors, distances);

// Print information about the neighbors found for the fifth point in the
// dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - " << neighbors[4].size() << " neighbors in distance range "
    << "[10.0, 50.0]." << std::endl;
if (neighbors[4].size() > 0)
{
  std::cout << " - First neighbor: index " << neighbors[4][0] << ", distance "
      << distances[4][0] << "." << std::endl;
}
```

---

Use an [Octree](../core/trees/octree.md) (a tree known to be faster in very few
dimensions) to perform range search in a tiny subset of the 3-dimensional LCDM
dataset.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::Load("lcdm_tiny.csv", dataset);

// Build a RangeSearch object on the LCDM dataset, and pass with `std::move()`
// so that we can avoid copying the dataset.  The search strategy is set to
// single-tree search.
mlpack::RangeSearch<mlpack::EuclideanDistance, arma::mat, mlpack::Octree> rs(
    std::move(dataset));

// Now search for any points with distance greater than 175.0.
std::vector<std::vector<double>> distances;
std::vector<std::vector<size_t>> neighbors;
rs.Search(mlpack::Range(175.0, DBL_MAX), neighbors, distances);

// Compute the maximum number of neighbors that any query point has, and the
// average.
size_t maxNeighbors = 0;
size_t sumNeighbors = 0;
for (size_t i = 0; i < neighbors.size(); ++i)
{
  maxNeighbors = std::max(maxNeighbors, neighbors[i].size());
  sumNeighbors += neighbors[i].size();
}
std::cout << "The maximum number of points with distance greater than 175.0 "
    << "that any query point has is " << maxNeighbors << "." << std::endl;
std::cout << "The average number of neighbors with distance greater than 175.0 "
    << "that a query point has is " << double(sumNeighbors) / neighbors.size()
    << "." << std::endl;
```

---

Use the [cover tree](../core/trees/cover_tree.md) to perform range search on the
`cloud` dataset using the Manhattan distance, with 32-bit floats used to
represent the data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::Load("cloud.csv", dataset);

// Construct the RangeSearch object using ManhattanDistance as the DistanceType.
mlpack::RangeSearch<mlpack::ManhattanDistance,
                    arma::fmat,
                    mlpack::StandardCoverTree> rs(std::move(dataset));

std::vector<std::vector<float>> distances;
std::vector<std::vector<size_t>> neighbors;

// Compute neighbors within the distance range [10.0, 100.0].
rs.Search(mlpack::RangeType<float>(10.0, 100.0), neighbors, distances);

// Print information about the neighbors found for the fifth point in the
// dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - " << neighbors[4].size() << " neighbors in distance range "
    << "[10.0, 100.0]." << std::endl;
if (neighbors[4].size() > 0)
{
  std::cout << " - First neighbor: index " << neighbors[4][0] << ", distance "
      << distances[4][0] << "." << std::endl;
}
```
