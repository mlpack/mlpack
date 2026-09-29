## `DBSCAN`

The `DBSCAN` class implements the clustering technique DBSCAN ("Density Based
Spatial Clustering of Applications with Noise").  It iteratively finds localized
high-density data regions by using range searches.  Nearby points in connected
high-density regions are grouped into clusters.  Clusters produced by DBSCAN may
have arbitrary shapes, and points far away from any high-density region will be
separately classified as noise.

DBSCAN does not require the user to guess the number of clusters, and
does not make any assumptions on the shape of the data.  Arbitrary distance
metrics are supported via template parameters.

#### Simple usage example:

```c++
// Use DBSCAN to cluster random data and print the number of points that
// fall into each cluster.

// Create random dataset with two separated 10-dimensional Gaussians.
arma::mat dataset = arma::join_rows(
    arma::randn<arma::mat>(10, 1000) - 3.0,  // 1000 points from N(-3, 1).
    arma::randn<arma::mat>(10, 1000) + 3.0,  // 1000 points from N( 3, 1).
    arma::randn<arma::mat>(10, 1) + 20.0);   // One outlier "noise" point.

// Step 1: create object.
mlpack::DBSCAN dbscan(5.0 /* radius */, 10 /* minPoints */);

// Step 2: perform clustering.
arma::Row<size_t> assignments;
arma::mat centroids;
dbscan.Cluster(dataset, assignments, centroids);

// Print the number of clusters.
std::cout << "Found " << centroids.n_cols << " centroids." << std::endl;

// Print the number of points in each cluster.
for (size_t c = 0; c < centroids.n_cols; ++c)
{
  std::cout << " * Cluster " << c << " has " << arma::accu(assignments == c)
      << " points." << std::endl;
}

// Print the number of noise points.
std::cout << " * " << arma::accu(assignments == SIZE_MAX) << " points "
    << "classified as noise." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `DBSCAN` objects.
 * [`Cluster()`](#clustering): perform clustering.
 * [Other functionality](#other-functionality) for loading, saving and
   inspecting.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [mlpack clustering algorithms](../modeling.md#clustering)
 * [DBSCAN on Wikipedia](https://en.wikipedia.org/wiki/DBSCAN)
 * [A density-based algorithm for discovering clusters in large spatial databases with noise (pdf)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)

### Constructors

 * `dbscan = DBSCAN(radius=0.5, minPoints=5, batchMode=true)`
   - Create a `DBSCAN` object with the specified parameters.
   - Clustering results are highly sensitive to the values of `radius` and
     `minPoints`; it is recommended to tune these parameters for your dataset!
     * See the notes below for more information on choosing these parameters.
   - mlpack's default [kd-tree](../core/trees/kdtree.md) dual-tree range search
     functionality will be used during [clustering](#clustering) for range
     search operations.

---

 * `dbscan = DBSCAN(radius, minPoints, batchMode, rangeSearch)`
 * `dbscan = DBSCAN(radius, minPoints, batchMode, rangeSearch, pointSelector)`
   - Create a `DBSCAN` object with the specified parameters, giving
     pre-instantiated
     [`RangeSearch` and `OrderedPointSelection` objects](#advanced-functionality-template-parameters) that will be used during [clustering](#clustering).
   - This overload is generally only useful if you want to reuse a `RangeSearch`
     object from elsewhere.

---

 * `dbscan = DBSCAN<RangeSearchType>(radius=0.5, minPoints=5, batchMode=true)`
 * `dbscan = DBSCAN<RangeSearchType>(radius, minPoints, batchMode, rangeSearch)`
   - Create a `DBSCAN` object with the specified parameters, giving a
     pre-instantiated `RangeSearchType` object that will be used during
     [clustering](#clustering).
   - The `RangeSearchType` template parameter can be arbitrarily chosen and is
     described in the
     [advanced functionality section](#advanced-functionality-template-parameters).

---

 * `dbscan = DBSCAN<RangeSearchType, PointSelectionPolicy>(radius=0.5, minPoints=5, batchMode=true)`
 * `dbscan = DBSCAN<RangeSearchType, PointSelectionPolicy>(radius, minPoints, batchMode, rangeSearch)`
 * `dbscan = DBSCAN<RangeSearchType, PointSelectionPolicy>(radius, minPoints, batchMode, rangeSearch, pointSelector)`
   - Create a `DBSCAN` object with the specified parameters, giving
     pre-instantiated `RangeSearchType` and `PointSelectionPolicy` objects that
     will be used during [clustering](#clustering).
   - The `RangeSearchType` and `PointSelectionPolicy` template parameters can be
     arbitrarily chosen and are described in the
     [advanced functionality section](#advanced-functionality-template-parameters).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `radius` | `double` | Maximum distance between points that are a part of the same cluster. | `0.5` |
| `minPoints` | `size_t` | Minimum number of points within distance `radius` for a point to be considered a 'core point' (e.g. the root of a cluster). | `5` |
| `batchMode` | `bool` | Whether to use batch-mode range search to find neighbors of points. | `true` |
| `rangeSearch` | [`RangeSearchType`](#advanced-functionality-template-parameters) | Instantiated object to perform range searches with. | `RangeSearchType()` |
| `pointSelector` | [`PointSelectionPolicy`](#advanced-functionality-template-parameters) | Instantiated object to select the next point to use as a candidate core point of a new cluster. | `PointSelectionPolicy()` |

***Notes:***

 - Clustering results are ***very*** sensitive to the settings of `radius` and
   `minPoints`!  The defaults for both of those are likely not correct for any
   dataset; ***manual tuning and experimentation is generally necessary***.

 - If `radius` is too small, then no points will be considered a part of the
   same cluster and all points will be classified as noise.  If `radius` is too
   large, then all points will be classified as one cluster.
   * One very simple way to find a starting point for the radius is to find the
     nearest neighbor of a subset of points in `data` using [`KNN`](knn.md), and
     then taking `2 * mean(distances)` as a starting point for `radius`.  (This
     is just a heuristic to give a _starting point_ for tuning.)

 - `minPoints` specifies the minimum number of neighboring points that a point
   must have to be the root of a cluster (e.g. a 'core point').  As this
   increases, the minimum number of points in a cluster also increases, but
   fewer points can be 'core points' that are the root of clusters.

 - Setting `batchMode` to `false` can keep memory usage lower, but at the
   potential cost of runtime slowdown.

 - By default, clustering is performed using the Euclidean distance.  A
   different or custom distance metric is specified by modifying the
   [`RangeSearchType`](#advanced-functionality-template-parameters) template
   parameter.  See also [an example](#custom-distance-metric-example).

### Clustering

 * `dbscan.Cluster(data, centroids)`
   - Cluster the given data, storing the resulting cluster centroids in
     `centroids`.
   - `centroids` will be set to size `data.n_rows` x `numClusters`, where
     `numClusters` is the number of clusters found by DBSCAN.
   - The `i`th cluster centroid can be obtained with `clusters.col(i)`.

 * `dbscan.Cluster(data, assignments)`
   - Cluster the given data, storing the resulting point assignments in
     `assignments`.
   - `assignments` will be set to size `data.n_cols`.
   - The cluster assignment of the point `data.col(i)` can be obtained with
     `assignments[i]`.
   - If the point `data.col(i)` has been classified as noise, then
     `assignments[i]` will be set to `SIZE_MAX`.  Otherwise, `assignments[i]`
     will take values in the range `[0, numClusters - 1]`.

 * `dbscan.Cluster(data, assignments, centroids)`
   - Cluster the given data, storing the resulting cluster centroids in
     `centroids` and the resulting point assignments in `assignments`.
   - `centroids` will be set to size `data.n_rows` x `numClusters`, where
     `numClusters` is the number of clusters found by DBSCAN.
   - The `i`th cluster centroid can be obtained with `clusters.col(i)`.
   - `assignments` will be set to size `data.n_cols`.
   - The cluster assignment of the point `data.col(i)` can be obtained with
     `assignments[i]`.
   - If the point `data.col(i)` has been classified as noise, then
     `assignments[i]` will be set to `SIZE_MAX`.  Otherwise, `assignments[i]`
     will take values in the range `[0, numClusters - 1]`.

---

#### Clustering Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix holding the dataset to be clustered. | _(N/A)_ |
| `centroids` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix that centroids will be stored into. | _(N/A)_ |
| `assignments` | [`arma::Row<size_t>`](../matrices.md) | Vector to store cluster assignments for each point into. | _(N/A)_ |

***Notes***:

 * Different types can be used for `data` and `centroids` (e.g., `arma::fmat` or
   any dense matrix type implementing the Armadillo API) by specifying a custom
   [`RangeSearchType`](#advanced-functionality-template-parameters).

### Other Functionality

 * A `DBSCAN` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

 * As an alternative to constructor parameters,
   - `radius` can be set with `dbscan.Radius(newRadius)`,
   - the minimum number of points for a core point can be set with
     `dbscan.MinPoints(newMinPoints)`, and
   - the batch mode setting can be set with `dbscan.BatchMode(newBatchMode)`.

 * `dbscan.RangeSearch()` returns a reference to the instantiated
    [`RangeSearchType`](#advanced-functionality-template-parameters) object used
    for range searching.

 * `dbscan.PointSelector()` returns a reference to the instantiated
    [`PointSelectionPolicy`](#advanced-functionality-template-parameters) object
    used to choose the first point of a new cluster.

### Simple Examples

Perform DBSCAN clustering on the satellite dataset and print the indices of any
noise point, as well as the average distance from each point to its assigned
centroid.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

// Create DBSCAN object with parameters tuned to the satellite dataset and
// perform clustering.
mlpack::DBSCAN dbscan(55.0 /* radius */, 3 /* minPoints */);
arma::mat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

// Print the number of clusters.
std::cout << "DBSCAN computed " << centroids.n_cols << " clusters."
    << std::endl;

// Compute the average distance from each point to its assigned centroid.
double sumDist = 0.0;
for (size_t i = 0; i < dataset.n_cols; ++i)
{
  if (assignments[i] != SIZE_MAX) // Filter out noise points.
  {
    sumDist += mlpack::EuclideanDistance::Evaluate(
        dataset.col(i), centroids.col(assignments[i]));
  }
  else
  {
    std::cout << " - Point " << i << " classified as noise." << std::endl;
  }
}
const double avgDist = sumDist / (double) dataset.n_cols;

std::cout << "Average distance from a point to its assigned centroid: "
    << avgDist << "." << std::endl;
```

---

Perform DBSCAN clustering on the wave energy farm dataset, setting `batchMode`
to `false` to save RAM usage during range searching.

```c++
// See https://datasets.mlpack.org/wave_energy_farm_100.csv.
arma::mat dataset;
mlpack::Load("wave_energy_farm_100.csv", dataset, mlpack::Fatal);

// Create DBSCAN object and set parameters.
mlpack::DBSCAN dbscan(10000.0 /* radius */,
                      10 /* minPoints */,
                      false /* batchMode */);

// Perform the clustering.
arma::mat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

std::cout << "DBSCAN found " << centroids.n_cols << " clusters."
    << std::endl;
std::cout << arma::accu(assignments == SIZE_MAX) << " points were classified "
    << "as noise." << std::endl;

// Save the centroids to disk.
mlpack::Save("wave_energy_centroids.csv", centroids);
```

---

<a name="custom-distance-metric-example">
Perform DBSCAN clustering on the cloud dataset using the Manhattan distance.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::Load("cloud.csv", dataset, mlpack::Fatal);

// Create the DBSCAN object using a custom distance type.
// Because DBSCAN uses RangeSearch for all of its distance-related computations,
// the distance metric is specified as part of the RangeSearchType template
// parameter.
mlpack::DBSCAN<mlpack::RangeSearch<mlpack::ManhattanDistance>> dbscan(
    50.0 /* radius */, 10 /* minPoints */);

// Perform clustering.
arma::mat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

// Print the number of clusters and the number of points in each cluster.
std::cout << "DBSCAN found " << centroids.n_cols << " clusters."
    << std::endl;
for (size_t i = 0; i < centroids.n_cols; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
std::cout << " - " << arma::accu(assignments == SIZE_MAX) << " points were "
    << "classified as noise and not assigned to any cluster." << std::endl;
```

---

Perform DBSCAN clustering on the cloud dataset, using 32-bit floating point
matrices to represent the data via the
[`RangeSearchType` template parameter](#advanced-functionality-template-parameters).

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::Load("cloud.csv", dataset, mlpack::Fatal);

// Create the DBSCAN object using a custom `RangeSearch` that uses `arma::fmat`
// as the matrix type.
using RangeSearchType = mlpack::RangeSearch<mlpack::EuclideanDistance,
                                            arma::fmat>;
mlpack::DBSCAN<RangeSearchType> dbscan(40.0 /* radius */, 10 /* minPoints */);

// Perform clustering.
arma::fmat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

// Print the number of clusters and the number of points in each cluster.
std::cout << "DBSCAN found " << centroids.n_cols << " clusters."
    << std::endl;
for (size_t i = 0; i < centroids.n_cols; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
std::cout << " - " << arma::accu(assignments == SIZE_MAX) << " points were "
    << "classified as noise and not assigned to any cluster." << std::endl;
```

---

Perform DBSCAN clustering on the cloud dataset using the
[L1 (Manhattan) distance](../core/distances.md#lmetric),
using mlpack's `RangeSearch` class with the
[`CoverTree`](../core/trees/cover_tree.md) for range search operations.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::Load("cloud.csv", dataset, mlpack::Fatal);

// Create the DBSCAN object using a custom `RangeSearch` that uses `arma::fmat`
// as the matrix type and `CoverTree` as the tree type.
using RangeSearchType = mlpack::RangeSearch<mlpack::ManhattanDistance,
                                            arma::fmat,
                                            mlpack::StandardCoverTree>;
mlpack::DBSCAN<RangeSearchType> dbscan(50.0 /* radius */, 10 /* minPoints */);

// Perform clustering.
arma::fmat centroids;
arma::Row<size_t> assignments;
dbscan.Cluster(dataset, assignments, centroids);

// Print the number of clusters and the number of points in each cluster.
std::cout << "DBSCAN found " << centroids.n_cols << " clusters."
    << std::endl;
for (size_t i = 0; i < centroids.n_cols; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
std::cout << " - " << arma::accu(assignments == SIZE_MAX) << " points were "
    << "classified as noise and not assigned to any cluster." << std::endl;
```

### Advanced Functionality: Template Parameters

The `DBSCAN` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```
DBSCAN<RangeSearchType, PointSelectionPolicy>
```

Arbitrary distance metrics are supported; as all distance-related computations
are performed by the `RangeSearchType` class, the distance metric is specified
as a part of that template parameter.

---

<!-- TODO: elaborate here once RangeSearch is documented -->

 * `RangeSearchType` specifies the algorithm to be used when performing range
   searches.
   - By default, the `RangeSearch` class is used, which uses an efficient
     dual-tree [`KDTree`](../core/trees/kdtree.md)-based search.

   - When `batchMode` is set to `false`, then single-tree search is used.

   - The `RangeSearch` class is itself configurable with template parameters;
     its full signature is:

```
RangeSearch<DistanceType, MatType, TreeType>
```

 * When using mlpack's `RangeSearch` class as `RangeSearchType`, each of its
   individual template parameters can be specified:
   - `DistanceType` can be any valid [distance metric](../core/distances.md);
     the default is [`EuclideanDistance`](../core/distances.md#lmetric).
   - `MatType` should be any matrix type implementing the Armadillo API; the
     default is [`arma::mat`](../matrices.md).  Other options include, e.g.,
     `arma::fmat`, and `arma::hmat`.
     * The `MatType` used here will be the same type that is accepted by
       [`Cluster()`](#clustering).
   - `TreeType` is the [tree type](../core/trees.md) used for tree-based
     searching.  By default, [`KDTree`](../core/trees/kdtree.md) is used.

 * To use a custom distance metric with `DBSCAN` without modifying any other
   template parameters, use `RangeSearch<DistanceType>`.
   - A list of distance metrics included with mlpack can be found
     [here](../core/distances.md).
   - For example, using the Manhattan distance could be done by specifying
     `RangeSearchType` as `RangeSearch<ManhattanDistance>`.

 * An entirely custom `RangeSearchType` must implement two typedefs and three
   member functions:

```c++
class CustomRangeSearchType
{
  // This typedef is what DBSCAN uses as `MatType`.
  using Mat = arma::mat; /* or any other Armadillo-compatible choice */
  // This typedef defines the element type that the matrix holds.
  using ElemType = typename Mat::elem_type;

  /**
   * Prepare for range search queries, using `referenceSet` as the set that is
   * being searched in.
   */
  void Train(const Mat& referenceSet);

  /**
   * This will always be called after a call to `Train()`.
   *
   * For each point in `querySet`, find *all* points in `referenceSet` within a
   * distance of `range.Lo()` and `range.Hi()`.  Store the indices of these
   * points in `neighbors` and their distances in `distances`.
   *
   * After the function is done, `neighbors[i][j]` should contain the index of
   * the `j`'th point in `referenceSet` that is within `range` of the point
   * `querySet.col(i)`.
   */
  void Search(const Mat& querySet,
              const RangeType<ElemType>& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<ElemType>>& distances);

  /**
   * This will always be called after a call to `Train()`.
   *
   * For each point in `referenceSet` (which was passed to `Train()`), find all
   * points within a distance of `range.Lo()` and `range.Hi()`.  Store the
   * indices of these points in `neighbors` and their distances in `distances`.
   *
   * After the function is done, `neighbors[i][j]` should contain the index of
   * the `j`'th point that is within `range` of the point `referenceSet.col(i)`.
   *
   * `neighbors[i]` should *not* contain `i`; that is, a point should not be
   * returned in its own set of neighbors, even if `range.Lo() == 0`.
   */
  void Search(const Mat& querySet,
              const RangeType<ElemType>& range,
              std::vector<std::vector<size_t>>& neighbors,
              std::vector<std::vector<ElemType>>& distances);
};
```

Note that it is generally easier to use a variant of mlpack's existing
`RangeSearch` class instead of implementing an entirely new one from scratch!

---

 * `PointSelectionPolicy` specifies the order in which points are selected as
   candidate roots of clusters.
   - By default, the `OrderedPointSelection` class is used, and is generally
     sufficient for all DBSCAN clustering tasks.  This selects the lowest-index
     unvisited point.

   - The `RandomPointSelection` class is also available; this selects randomly
     from the set of unvisited points.

   - DBSCAN point cluster assignment is greedy; a point is assigned to the first
     cluster it is within a distance of `radius` of.  Therefore, to some
     limited extent, clustering behavior can be controlled by
     `PointSelectionPolicy`.

   - A custom point selection strategy must implement one member function:

```c++
class CustomPointSelection
{
 public:
  /**
   * Select the next point to use as a candidate cluster root for DBSCAN.  This
   * method should return the index of the point in `data` to use.
   *
   * MatType is the Armadillo-compatible matrix type used to store the data
   * points.
   *
   * @param numVisited Number of points that have been visited so far.
   * @param visited Bitset indicating which points have already been visited.
   *      Note that more than `numVisited` points may have been visited, since
   *      any point's neighbors are considered 'visited' at each iteration.
   * @param data Matrix of data points.
   */
  template<typename MatType>
  static size_t Select(const size_t numVisited,
                       const std::vector<bool>& visited,
                       const MatType& data);
};
```
