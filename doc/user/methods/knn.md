## `KNN`

The `KNN` class implements `k`-nearest neighbor search (both exact and
approximate), a core computational task that is useful in many machine learning
situations.  mlpack's `KNN` class uses [trees](../core/trees.md), by default the
[`KDTree`](../core/tree/kdtree.md), to provide significantly accelerated
computation; depending on input options, an efficient single-tree or dual-tree
algorithm is used.

mlpack's `KNN` implementation builds a tree structure on the dataset to be
searched (called the 'reference set').  This provides significant efficiency
gains, and is most effective in low to medium dimensions (typically less than
100).

<!-- TODO: link to LSH as an alternative for higher dimensions -->

`KNN` can either search for the nearest neighbors of all points in the reference
set (also called all-nearest-neighbors), or `KNN` can search for the nearest
neighbors in the reference set of a different set of query points.

The `KNN` class supports configurable behavior, with numerous runtime and
compile-time parameters, including the distance metric, type of data, search
strategy, and tree type.

Note that the `KNN` class is not a classifier, and instead focuses on simply
computing the nearest neighbors of points.

#### Simple usage example:

```c++
// Compute the 5 exact nearest neighbors of every point of random numeric data.

// All data is uniform random: 10-dimensional data.  Replace with a data::Load()
// call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.

mlpack::KNN knn;                     // Step 1: create object.
knn.Train(dataset);                  // Step 2: set the dataset to search in.
arma::mat distances;
arma::Mat<size_t> neighbors;
knn.Search(5, neighbors, distances); // Step 3: find 5 nearest neighbors of
                                     //         every point in `dataset`.

// Print some information about the results.
std::cout << "Found " << neighbors.n_rows << " neighbors for each of "
    << neighbors.n_cols << " points in the dataset." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `KNN` objects.
 * [Search modes](#search-modes): details of search strategies supported by
   `KNN`.
 * [Setting the reference set (`Train()`)](#setting-the-reference-set-train):
   set the dataset that will be searched for nearest neighbors.
 * [Searching for neighbors](#searching-for-neighbors): call `Search()` to
   compute nearest neighbors (exact or approximate).
 * [Computing quality metrics](#computing-quality-metrics) to determine how
   accurate the computed nearest neighbors are, if approximate search was used.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for
   configuring search strategy behavior, using different element types, and
   using different [distance metrics](../core/distances.md).
 * [Advanced examples](#advanced-examples) that make use of custom template
   parameters.

#### See also:

 * [mlpack trees](../core/trees.md)
 * [mlpack geometric algorithms](../modeling.md#geometric-algorithms)
 * [Nearest neighbor search on Wikipedia](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
 * [Tree-Independent Dual-Tree Algorithms (pdf)](https://www.ratml.org/pub/pdf/2013tree.pdf)

### Constructors

 * `knn = KNN()`
 * `knn = KNN(mode=DUAL_TREE_MODE, epsilon=0)`
   - Construct a `KNN` object, optionally using the given `mode` for search and
     `epsilon` for approximation level.
   - This does not set the reference set to be searched!
     [`Train()`](#setting-the-reference-set-train) must be called before calling
     [`Search()`](#searching-for-neighbors).

 * `knn = KNN(referenceSet)`
 * `knn = KNN(referenceSet, mode=DUAL_TREE_MODE, epsilon=0)`
   - Construct a `KNN` object on the given set of reference points, using the
     given `mode` for search and `epsilon` for approximation level.
   - This will build a [`KDTree`](../trees/kdtree.md) with default parameters on
     `referenceSet`, if `mode` is not `NAIVE_MODE`.
   - If `referenceSet` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceSet)`) to avoid copying `referenceSet`.  The dataset
     will still be accessible via [`ReferenceSet()`](#other-functionality), but
     points may be in shuffled order.

 * `knn = KNN(referenceTree)`
 * `knn = KNN(referenceTree, mode=DUAL_TREE_MODE, epsilon=0)
   - Construct a `KNN` object with a pre-built tree `referenceTree`, which
     should be of type `KNN::Tree` (a convenience typedef of
     [`KDTree`](../core/trees/kdtree.md) that uses
     [`NeighborSearchStat`](../core/trees/binary_space_tree.md#neighborsearchstat)
     as its
     [`StatisticType`](../core/trees/binary_space_tree.md#statistictype)).
   - The search mode will be set to `mode` and approximation level will be set
     to `epsilon`.
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
| `referenceTree` | `KNN::Tree` (a [`KDTree`](../core/tree/kdtree.md)) | Pre-built kd-tree on reference data. | _(N/A)_ |
| `mode` | `enum NeighborSearchMode` | The search technique that will be used when `Search()` is called.  Must be one of `NAIVE_MODE`, `SINGLE_TREE_MODE`, `DUAL_TREE_MODE`, or `GREEDY_SINGLE_TREE_MODE`.  [More details.](#search-mode) | `DUAL_TREE_MODE` |
| `epsilon` | `double` | Allowed relative approximation error.  `0` means exact search.  Must be non-negative. | `0.0` |

***Notes:***

 - By default, exact nearest neighbors are found.  Set `epsilon` to a positive
   value to enable approximation (higher `epsilon` means more approximation is
   allowed).

 - If constructing a tree manually, the `KNN::Tree` type can be used (e.g.,
   `tree = KNN::Tree(referenceData)`.  `KNN::Tree` is a convenience typedef of
   either [`KDTree`](../doc/user/core/tree/kdtree.md) or the chosen `TreeType`
   if [custom template parameters](#advanced-functionality-template-parameters)
   are being used.

### Search modes

The `KNN` class can search for nearest neighbors using one of the following four
strategies.  These can be specified in the constructor as the `mode` parameters,
or by calling `knn.SearchMode() = mode`.

 * `DUAL_TREE_MODE` _(default)_: two trees will be used at search time with a
   [dual-tree algorithm](https://ratml.org/pub/pdf/2013tree.pdf) to allow the
   maximum amount of pruning.
   - This is generally the fastest strategy.
   - Under some assumptions on the structure of the dataset and the tree type
     being used, dual-tree search
     [scales linearly](https://ratml.org/pub/pdf/2015plug.pdf) (e.g. `O(1)` time
     for each point whose nearest neighbors are being computed).
   - Backtracking search is performed to find either exact nearest neighbors or
     approximate nearest neighbors if `knn.Epsilon() > 0`.

 * `SINGLE_TREE_MODE`: a tree built on the reference points will be traversed
   once for each point whose nearest neighbors are being searched for.
   - Single-tree search generally empirically
     [scales logarithmically](https://en.wikipedia.org/wiki/Nearest_neighbor_search#Space_partitioning).
   - Backtracking search is performed to find either exact nearest neighbors or
     approximate nearest neighbors if `knn.Epsilon() > 0`.

 * `GREEDY_SINGLE_TREE_MODE`: for each point whose nearest neighbors are being
   searched for, a tree built on the reference points will be traversed in a
   greedy manner---recursing directly and only to the nearest node in the tree
   to find nearest neighbor candidates.
   - The approximation level with this strategy cannot be controlled; the
     setting of `knn.Epsilon()` is ignored.
   - Greedy single-tree search scales logarithmically (e.g. `O(log N)` for each
     point whose neighbors are being computed, if the size of the reference set
     is `N`); however, since no backtracking is performed, results are obtained
     extremely efficiently.
   - This strategy is most effective when
     [spill trees](../../core/trees/sp_tree.md)] are used; to do this, use
     `SPTree` or another [spill tree variant](../../core/trees/spill_tree.md) as
     the
     [`TreeType` template parameter](#advanced-functionality-template-parameters).

 * `NAIVE_MODE`: brute-force search---for each point whose nearest neighbors are
   being searched for, compute the distance to *every* point in the reference
   set.
   - This strategy gives exact results always; the setting of `knn.Epsilon()` is
     ignored.
   - Brute-force search scales poorly, with a runtime cost of `O(N)` (where `N`
     is the size of the reference set) per point.
   - However, brute-force search does not suffer from
     [poor performance in high dimensions](https://en.wikipedia.org/wiki/K-d_tree#Degradation_in_performance_with_high-dimensional_data) as trees often do.
   - When this strategy is used, no tree structure is used.

### Setting the reference set (`Train()`)

If the reference set was not set in the constructor, or if it needs to be
changed to a new reference set, the `Train()` method can be used.

 * `knn.Train(referenceSet)`
   - Set the reference set to `referenceSet`.
   - This will build a [`KDTree`](../trees/kdtree.md) with default parameters on
     `referenceSet`, if `mode` is not [`NAIVE_MODE`](#search-modes).
   - If `referenceSet` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceSet)`) to avoid copying `referenceSet`.  The dataset
     will still be accessible via [`ReferenceSet()`](#other-functionality), but
     points may be in shuffled order.

 * `knn.Train(referenceTree)`
   - Set the reference tree to `referenceTree`, which should be of type
     `KNN::Tree` (a convenience typedef of [`KDTree`](../core/trees/kdtree.md)
     that uses
     [`NeighborSearchStat`](../core/trees/binary_space_tree.md#neighborsearchstat)
     as its
     [`StatisticType`](../core/trees/binary_space_tree.md#statistictype)).
   - If `referenceTree` is not needed elsewhere, pass with `std::move()` (e.g.
     `std::move(referenceTree)`) to avoid copying `referenceTree`.  The tree
     will still be accessible via [`ReferenceTree()`](#other-functionality).

### Searching for neighbors

Once the reference set and parameters are set, searching for nearest neighbors
can be done with the `Search()` method.

 * `knn.Search(k, neighbors, distances)`
   - Search for the `k` nearest neighbors of all points in the reference set
     (e.g. [`knn.ReferenceSet()`](#other-functionality)), storing the results in
     `neighbors` and `distances`.
   - `neighbors` and `distances` will be set to have `k` rows and
     `knn.ReferenceSet().n_cols` columns.
   - `neighbors(i, j)` (e.g. the `i`th row and `j`th column of `neighbors`) will
     hold the column index of the `i`th nearest neighbor of the `j`th point in
     `knn.ReferenceSet()`.
   - That is, the `i`th nearest neighbor of `knn.ReferenceSet().col(j)` is
     `knn.ReferenceSet().col(neighbors(i, j))`.
   - `distances(i, j)` will hold the distance between the `j`th point in
     `knn.ReferenceSet()` and its `i`th nearest neighbor.

 * `knn.Search(querySet, k, neighbors, distances)`
   - Search for the `k` nearest neighbors in the reference set of all points in
     `querySet`, storing the results in `neighbors` and `distances`.
   - `neighbors` and `distances` will be set to have `k` rows and
     `querySet.n_cols` columns.
   - `neighbors(i, j)` (e.g. the `i`th row and `j`th column of `neighbors`) will
     hold the column index in `knn.ReferenceSet()` of the `i`th nearest neighbor
     of the `j`th point in `querySet`.
   - That is, the `i`th nearest neighbor of `querySet.col(j)` is
     `knn.ReferenceSet().col(neighbors(i, j))`.
   - `distances(i, j)` will hold the distance between the `j`th point in
     `querySet` and its `i`th nearest neighbor in `knn.ReferenceSet()`.

 * `knn.Search(queryTree, k, neighbors, distances, sameSet=false)`
   - Search for the `k` nearest neighbors in the reference set of all points in
     `queryTree`, storing the result in `neighbors` and `distances`.
   - `neighbors` and `distances` will be set to have `k` rows and
     `queryTree.Dataset().n_cols` columns.
   - `neighbors(i, j)` (e.g. the `i`th row and `j`th column of `neighbors`) will
     hold the column index in `knn.ReferenceSet()` of the `i`th nearest neighbor
     of the `j`th point in `queryTree.Dataset()`.
   - That is, the `i`th nearest neighbor of `queryTree.Dataset().col(j)` is
     `knn.ReferenceSet().col(neighbors(i, j))`.
   - `distances(i, j)` will hold the distance between the `j`th point in
     `queryTree.Dataset()` and its `i`th nearest neighbor in
     `knn.ReferenceSet()`.

***Notes***:

 * When `querySet` and `queryTree` are not specified, a point will not return
   itself as its nearest neighbor.  However, if there are duplicate points `x`
   and `y` in the dataset, `y` may be returned as the nearest neighbor of `x`.

 * If `knn.Epsilon() > 0` and the search mode is
   [`DUAL_TREE_MODE` or `SINGLE_TREE_MODE`](#search-modes), then the search will
   return approximate nearest neighbors within a relative distance of
   `knn.Epsilon()` of the true nearest neighbor.

 * `knn.Epsilon()` is ignored when the search mode is
   [`GREEDY_SINGLE_TREE_MODE` or `NAIVE_MODE`](#search-modes).

 * When using a `queryTree` multiple times, the bounds in the tree must be
   reset.  Call `knn.ResetTree(queryTree)` after each call to `Search()` to
   reset the bounds, or call `node.Stat().Reset()` on each node in `queryTree`.

---

#### Search Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `querySet` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix of query points for which the nearest neighbors in the reference set should be found. |
| `k` | `size_t` | Number of nearest neighbors to search for. |
| `neighbors` | [`arma::Mat<size_t>`](../matrices.md) | Matrix to store indices
of nearest neighbors into.  Will be set to size `k` x `N`, where `N` is the number of points in the query set (if specified), or the reference set (if not). |
| `distances` | [`arma::mat`](../matrices.md) | Matrix to store distances to
nearest neighbors into.  Will be set to the same size as `neighbors`. |

### Computing quality metrics

If approximate nearest neighbor search is performed (e.g. if `knn.Epsilon() >
0`), and exact nearest neighbors are known, it is possible to compute quality
metrics of the approximate search.

 * `double error = knn.EffectiveError(computedDistances, exactDistances)`
   - Given a matrix of exact distances and computed approximate distances, both
     with the same size (rows equal to `k`, columns equal to the number of
     points in the query or reference set), compute the average relative error
     of the computed distances.
   - `computedDistances` and `exactDistances` should be matrices produced by
     [`knn.Search()`](#searching-for-neighbors).
   - This will be no greater than [`knn.Epsilon()`](#other-functionality).

 * `double recall = knn.Recall(computedNeighbors, trueNeighbors)`
   - Given a matrix containing indices of exact nearest neighbors and computed
     approximate neighbors, both with the same size (rows equal to `k`, columns
     equal to the number of points in the query or reference set), compute the
     recall (percentage of true neighbors found).
   - `computedNeighbors` and `trueNeighbors` should be matrices produced by
     [`knn.Search()`](#searching-for-neighbors).
   - The recall will be between `0.0` and `1.0`, with `1.0` indicating perfect
     recall.

### Other functionality

 - `knn.ReferenceSet()` will return a `const arma::mat&` representing the data
   points in the reference set.  This matrix cannot be modified.
   * If a
     [custom `MatType` template parameter](#advanced-functionality-template-parameters)
     has been specified, then the return type will be `const MatType&`.

 - `knn.ReferenceTree()` will return a `KNN::Tree*` (a
   [`KDTree`](../../core/trees/kdtree.md) with
   [`NeighborSearchStat`](../../core/trees/binary_space_tree.md#neighborsearchstat)
   as the
   [`StatisticType`](../../core/trees/binary_space_tree.md#statistictype)).
   * This is the tree that will be used at search time, if the search mode is
     not [`NAIVE_MODE`](#search-modes).
   * If the search mode was [`NAIVE_MODE`](#search-modes) when the object was
     constructed, then `knn.ReferenceTree()` will return `nullptr`.
   * If a
     [custom `TreeType` template parameter](#advanced-functionality-template-parameters)
     has been specified, then `KNN::Tree*` will be that type of tree, not a
     `KDTree`.

 - `knn.SearchMode()` will return the [search mode](#search-modes) that will be
   used when `knn.Search()` is called.

 - `knn.SearchMode() = newMode` will set the [search mode](#search-modes) to
   `newMode`, which must be one of the supported search modes.

 - `knn.Epsilon()` returns a `double` representing the allowed level of
   approximation.  If `0` and `knn.SearchMode()` is either dual- or single-tree
   search, then `knn.Search()` will return exact results.
   * `knn.Epsilon() = eps` will set the allowed level of approximation to `eps`.
     `eps` must be greater than or equal to `0.0`.

 - After calling `knn.Search()`, `knn.BaseCases()` will return a `size_t`
   representing the number of point-to-point distance computations that were
   performed, if a [tree-traversing search mode](#search-modes) was used.

 - After calling `knn.Search()`, `knn.Scores()` will return a `size_t` indicates
   the number of tree nodes that were visited during search, if a
   [tree-traversing search mode](#search-modes) was used.

 - A `KNN` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).  Note
   that for large reference sets, this will also serialize the dataset
   (`knn.ReferenceSet()`) and the tree (`knn.Tree()`), and so the resulting file
   may be quite large.

 - `KNN::Tree` is a convenience typedef representing the type of the tree that
   is used for searching.
   * By default, this is a [`KDTree`](../../core/trees/kdtree.md); specifically:
     `KNN::Tree = KDTree<EuclideanDistance, NeighborSearchStat, arma::mat>`.
   * If a
     [custom `TreeType`, `DistanceType`, and/or `MatType`](#advanced-functionality-template-parameters)
     are specified, then
     `KNN<DistanceType, TreeType, MatType>::Tree = TreeType<DistanceType, NeighborSearchStat, MatType>`.
   * A custom tree can be built and passed to
     [`Train()`](#setting-the-reference-set-train) or the
     [constructor](#constructors) with, e.g., `tree = KNN::Tree(referenceSet)`
     or `tree = KNN::Tree(std::move(referenceSet))`.

### Simple examples

Find the exact nearest neighbor of every point in the `cloud` dataset.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Construct the KNN object; this will avoid copies via std::move(), and build a
// kd-tree on the dataset.
mlpack::KNN knn(std::move(dataset));

arma::mat distances;
arma::Mat<size_t> neighbors;

// Compute the exact nearest neighbor.
knn.Search(1, neighbors, distances);

// Print the nearest neighbor and distance of the fifth point in the dataset.
std::cout << "Point 4:" << std::endl;
std::cout << " - Point values: " << knn.ReferenceSet().col(4).t();
std::cout << " - Index of nearest neighbor: " << neighbors(0, 4) << "."
    << std::endl;
std::cout << " - Distance to nearest neighbor: " << distances(0, 4) << "."
    << std::endl;
```

---

Split the `corel-histogram` dataset into two sets, and find the exact nearest
neighbor in the first set of every point in the second set.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::data::Load("corel-histogram.csv", dataset);

// Split the dataset into two equal-sized sets randomly with `data::Split()`.
arma::mat referenceSet, querySet;
mlpack::data::Split(dataset, referenceSet, querySet, 0.5);

// Construct the KNN object, building a tree on the reference set.  Copies are
// avoided by the use of `std::move()`.
mlpack::KNN knn(std::move(referenceSet));

arma::mat distances;
arma::Mat<size_t> neighbors;

// Compute the exact nearest neighbor in `referenceSet` of every point in
// `querySet`.
knn.Search(querySet, 1, neighbors, distances);

// Print information about the dual-tree traversal.
std::cout << "Dual-tree traversal computed " << knn.BaseCases()
    << " point-to-point distances and " << knn.Scores()
    << " tree node-to-tree node distances." << std::endl;

// Print information about nearest neighbors of the points in the query set.
std::cout << "The nearest neighbor of query point 3 is reference point index "
    << neighbors(0, 3) << ", with distance " << distances(0, 3) << "."
    << std::endl;
std::cout << "The L2-norm of reference point " << neighbors(0, 3) << " is "
    << arma::norm(knn.ReferenceSet().col(neighbors(0, 3)), 2) << "."
    << std::endl;

// Compute the average nearest neighbor distance for all points in the query
// set.
const double averageDist = arma::mean(arma::vectorise(distances));
std::cout << "Average distance between a query point and its nearest neighbor: "
    << averageDist << "." << std::endl;
```

---

Perform approximate single-tree search to find 5 nearest neighbors of the first
point in a subset of the `LCDM` dataset.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::data::Load("lcdm_tiny.csv", dataset);

// Build a KNN object on the LCDM dataset, and pass with `std::move()` so that
// we can avoid copying the dataset.  Set the search mode to single-tree mode.
mlpack::KNN knn(std::move(dataset), mlpack::SINGLE_TREE_MODE);

// Now we will compute the 5 nearest neighbors of the first point in the
// dataset.
//
// NOTE: because the first point is in the reference set, and because we are
// passing a separate query set, KNN will return that the nearest neighbor of
// the point is itself!  This is an important caveat to be aware of when calling
// Search() with a query set.
arma::mat distances;
arma::Mat<size_t> neighbors;
knn.Search(knn.ReferenceSet().col(0), 5, neighbors, distances);

std::cout << "The five nearest neighbors of the first point in the LCDM dataset"
    << " are:" << std::endl;
for (size_t k = 0; k < 5; ++k)
{
  std::cout << " - " << neighbors(k, 0) << " (with distance " << distances(k, 0)
      << ")." << std::endl;
  if (k == 0)
  {
    std::cout << "    (the first point's nearest neighbor is itself, because it"
        << " is in the reference set, and we called Query() with a separate "
        << "query set!)" << std::endl;
  }
}
```

---

Use greedy single-tree search to find 5 approximate nearest neighbors of every
point in the `cloud` dataset.  Then, compute the exact nearest neighbors, and use these to find the average error and recall of the approximate search.

***Note***: greedy single-tree search is far more effective when using spill
trees---see the [advanced examples](#advanced-examples) for another example that
does exactly that.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Build a tree on the dataset and set the search mode to greedy single tree
// mode.
mlpack::KNN knn(std::move(dataset), mlpack::GREEDY_SINGLE_TREE_MODE);

// Compute the 5 approximate nearest neighbors of every point in the dataset.
arma::mat distances;
arma::Mat<size_t> neighbors;
knn.Search(5, neighbors, distances);

std::cout << "Greedy approximate kNN search computed " << knn.BaseCases()
    << " point-to-point distances and visited " << knn.Scores()
    << " tree nodes in total." << std::endl;

// Now switch to exact mode and compute the true neighbors and distances.
arma::Mat<size_t> trueNeighbors;
arma::mat trueDistances;
knn.SearchMode() = mlpack::DUAL_TREE_MODE;
knn.Epsilon() = 0.0;
knn.Search(5, trueNeighbors, trueDistances);

// Compute the recall and effective error.
const double recall = knn.Recall(neighbors, trueNeighbors);
const double effectiveError = knn.EffectiveError(distances, trueDistances);

std::cout << "Recall of greedy search: " << recall << "." << std::endl;
std::cout << "Effective error of greedy search: " << effectiveError << "."
    << std::endl;
```

---

Build a `KNN` object on the `cloud` dataset and save it to disk.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset);

// Build the tree.
mlpack::KNN knn(std::move(dataset));

// Save the KNN object to disk with the name 'knn'.
mlpack::data::Save("knn.bin", "knn", knn);

std::cout << "Successfully saved KNN model to 'knn.bin'." << std::endl;
```

---

Load a `KNN` object from disk, and inspect the
[`KDTree`](../../core/trees/kdtree.md) that is held in the object.

```c++
// Load the KNN object with name 'knn' from 'knn.bin'.
mlpack::KNN knn;
mlpack::data::Load("knn.bin", "knn", knn);

// Inspect the KDTree held by the KNN object.
std::cout << "The KDTree in the KNN object in 'knn.bin' holds "
    << knn.ReferenceTree().NumDescendants() << " points." << std::endl;
std::cout << "The root of the tree has " << knn.ReferenceTree().NumChildren()
    << " children." << std::endl;
if (knn.ReferenceTree().NumChildren() == 2)
{
  std::cout << " - The left child holds "
      << knn.ReferenceTree().Child(0).NumDescendants() << " points."
      << std::endl;
  std::cout << " - The right child holds "
      << knn.ReferenceTree().Child(1).NumDescendants() << " points."
      << std::endl;
}
```

---

Compute the 5 approximate nearest neighbors of two subsets of the `covertype`
dataset using a pre-built query tree.  Then, reuse the query tree to compute the
exact neighbors and compute the effective error.

```c++
// See https://datasets.mlpack.org/covertype.data.csv.
arma::mat dataset;
mlpack::data::Load("covertype.data.csv", dataset);

// Split the covertype dataset into two parts of equal size.
arma::mat referenceSet, querySet;
mlpack::data::Split(dataset, referenceSet, querySet, 0.5);

// Build the KNN object, passing the reference set with `std::move()` to avoid a
// copy.  We use the default dual-tree mode for search and set the maximum
// allowed relative error to 0.1 (10%).
mlpack::KNN knn(std::move(referenceSet), mlpack::DUAL_TREE_MODE, 0.1);

// Now build a tree on the query points.  This is a KDTree, and we manually
// specify a leaf size of 50 points.  Note that the KDTree rearranges the
// ordering of points in the query set.
mlpack::KNN::Tree queryTree(std::move(querySet));

// Compute the 5 approximate nearest neighbors of all points in the query set.
arma::mat distances;
arma::Mat<size_t> neighbors;
knn.Search(queryTree, 5, neighbors, distances);

// Now compute the exact neighbors---but since we are using dual-tree mode and
// an externally-constructed query tree, we must reset the bounds!
arma::mat trueDistances;
arma::Mat<size_t> trueNeighbors;
knn.ResetTree(queryTree);
knn.Epsilon() = 0;
knn.Search(queryTree, 5, trueNeighbors, trueDistances);

// Compute the effective error.
const double effectiveError = knn.EffectiveError(distances, trueDistances);

std::cout << "Effective error of approximate dual-tree search was "
    << effectiveError << " (limit via knn.Epsilon() was 0.1)." << std::endl;
```

---

### Advanced functionality: template parameters

The `KNN`

#### Custom distance metrics

#### Different element types

#### Custom tree types

#### Different traversal types

### Advanced examples
