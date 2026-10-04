## `KMeans`

The `KMeans` class implements the standard `k`-means clustering algorithm.
`k`-means clusters data by iteratively computing the centroids of each of the
`k` clusters, and then reassigning points to their closest centroid.  `k`-means
is probably the most widely used clustering technique; it requires a single
parameter, `k` (the number of clusters).

mlpack's `KMeans` class is highly optimized, supporting numerous different
strategies for accelerated computation.  Via template parameters, distance
metrics and other behaviors are configurable.

#### Simple usage example:

```c++
// Use k-means to cluster random data and print the number of points that fall
// into each cluster.

// Create random dataset with two separated 10-dimensional Gaussians.
arma::mat dataset = arma::join_rows(
    arma::randn<arma::mat>(10, 1000) + 3.0,  // 1000 points from N(-3, 1).
    arma::randn<arma::mat>(10, 1000) - 3.0); // 1000 points from N( 3, 1).

mlpack::KMeans km;                              // Step 1: create object.
arma::Row<size_t> assignments;
arma::mat centroids;
km.Cluster(dataset, 2, assignments, centroids); // Step 2: perform clustering.

// Print the number of points in each cluster.
for (size_t c = 0; c < centroids.n_cols; ++c)
{
  std::cout << " * Cluster " << c << " has " << arma::accu(assignments == c)
      << " points." << std::endl;
}
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `KMeans` objects.
 * [`Cluster()`](#clustering): perform clustering.
 * [Other functionality](#other-functionality) for loading, saving, inspecting,
   and estimating the readius to use.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.
 * [Advanced examples](#advanced-examples) that make use of template parameters.

#### See also:

 * [mlpack clustering algorithms](../modeling.md#clustering)
 * [k-means on Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
 * [`MeanShift`](mean_shift.md)
 * [`DBSCAN`](dbscan.md)

### Constructors

 * `KMeans()`
 * `KMeans(maxIterations=1000)`
   - Create a `KMeans` object, optionally specifying the maximum number of
     iterations.
   - ***Note:*** by default, `KMeans` uses brute-force computation of centroids
     and assignments; this is not generally the most efficient approach.  It is
     strongly recommended to use
     [template parameters](#advanced-functionality-template-parameters) to
     select the best algorithm for your data with the
     [`LloydStepType`](#lloydsteptype) parameter.

---

 * `KMeans<DistanceType>()`
 * `KMeans<DistanceType>(maxIterations=1000, distance=DistanceType())`
   - Create a `KMeans` object with the specified `DistanceType`.
   - Optionally, specify the maximum number of iterations and provide an
     instantiated distance metric.
   - See the [advanced functionality section](#advanced-functionality-template-parameters) for more details on custom distance metrics.

---

 * `KMeans<DistanceType, InitialPartitionPolicy>()`
 * `KMeans<DistanceType, InitialPartitionPolicy>(maxIterations=1000, distance=DistanceType(), partitioner=InitialPartitionPolicy())`
   - Create a `KMeans` object with the specified `DistanceType` and
     `InitialPartitionPolicy`.
   - Optionally, specify the maximum number of iterations and provide an
     instantiated distance metric and initial partition policy.
   - See the [advanced functionality section](#advanced-functionality-template-parameters) for more details on custom distance metrics and custom initial partition policies.

---

 * `KMeans<DistanceType, InitialPartitionPolicy, EmptyClusterPolicy>()`
 * `KMeans<DistanceType, InitialPartitionPolicy, EmptyClusterPolicy>(maxIterations=1000, distance=DistanceType(), partitioner=InitialPartitionPolicy(), emptyClusterAction=EmptyClusterPolicy())`
   - Create a `KMeans` object with the specified `DistanceType`,
     `InitialPartitionPolicy`, and `EmptyClusterPolicy`.
   - Optionally, specify the maximum number of iterations and provide an
     instantiated distance metric, initial partition policy, and empty cluster
     policy.
   - See the [advanced functionality section](#advanced-functionality-template-parameters) for more details on custom distance metrics, custom initial partition policies, and custom empty cluster actions.

---

 * `KMeans<DistanceType, InitialPartitionPolicy, EmptyClusterPolicy, LloydStepType>()`
 * `KMeans<DistanceType, InitialPartitionPolicy, EmptyClusterPolicy, LloydStepType>(maxIterations=1000, distance=DistanceType(), partitioner=InitialPartitionPolicy(), emptyClusterAction=EmptyClusterPolicy())`
   - Create a `KMeans` object with the specified `DistanceType`,
     `InitialPartitionPolicy`, and `EmptyClusterPolicy`.
   - Optionally, specify the maximum number of iterations and provide an
     instantiated distance metric, initial partition policy, and empty cluster
     policy.
   - See the [advanced functionality section](#advanced-functionality-template-parameters) for more details on custom distance metrics, custom initial partition policies, custom empty cluster actions, and custom step policies.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `maxIterations` | `size_t` | Maximum number of iterations of the mean shift algorithm to run. | `1000` |
| `distance` | `DistanceType` | Instantiated distance metric to use (only when specifying a [custom `DistanceType`](#advanced-functionality-template-parameters). | `DistanceType()` |
| `partitioner` | `InitialPartitionPolicy` | Instantiated object that computes initial cluster assignments (only for when specifying a [custom `InitialPartitionPolicy`](#advanced-functionality-template-parameters)). | `InitialPartitionPolicy()` |
| `emptyClusterAction` | `EmptyClusterPolicy` | Instantiated object that specifies the action to take when a cluster is empty at the end of an iteration (only for when specifying a [custom `EmptyClusterPolicy`](#advanced-functionality-template-parameters)). |

### Clustering

 * `km.Cluster(data, k, assignments, initialAssignmentGuess=false)`
   - Cluster the given data into `k` clusters, storing point assignments into
     the given `assignments` vector.
   - `assignments` will be set to length `data.n_cols`; the assignment of the
     `i`th point can be obtained with `assignments[i]`.
   - If `initialAssignmentGuess` is `true`, then `assignments` is expected to
     have length `data.n_cols` when `Cluster()` is called, and those assignments
     are used as the initial clustering.
   - Clustering will continue for up to `maxIterations` iterations, or until the
     cluster distortion (see notes below) drops below `1e-5`.

 * `km.Cluster(data, k, centroids, initialCentroidGuess=false)`
   - Cluster the given data into `k` clusters, storing computed centroids into
     the `centroids` matrix.
   - `centroids` will be set to size `data.n_rows` x `k`; the `i`th cluster
     centroid can be obtained with `centroids.col(i)`.
   - If `initialCentroidGuess` is `true`, then `centroids` is expected to have
     size `data.n_rows` x `k` when `Cluster()` is called, and those centroids
     are used as the initial clustering.
   - Clustering will continue for up to `maxIterations` iterations, or until the
     cluster distortion (see notes below) drops below `1e-5`.

 * `km.Cluster(data.k, assignments, centroids, initialAssignmentGuess=false, initialCentroidGuess=false)`
   - Cluster the given data into `k` clusters, storing point assignments into
     the given `assignments` vector, and computed centroids into the `centroids`
     matrix.
   - `assignments` will be set to length `data.n_cols`; the assignment of the
     `i`th point can be obtained with `assignments[i]`.
   - `centroids` will be set to size `data.n_rows` x `k`; the `i`th cluster
     centroid can be obtained with `centroids.col(i)`.
   - If `initialAssignmentGuess` is `true`, then `assignments` is expected to
     have length `data.n_cols` when `Cluster()` is called, and those assignments
     are used as the initial clustering.
   - If `initialCentroidGuess` is `true` and `initialAssignmentGuess` is
     `false`, then `centroids` is expected to have size `data.n_rows` x `k` when
     `Cluster()` is called, and those centroids are used as the initial
     clustering.
   - Clustering will continue for up to `maxIterations` iterations, or until the
     cluster distortion (see notes below) drops below `1e-5`.

---

#### Clustering Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix holding the dataset to be clustered. | _(N/A)_ |
| `k` | `size_t` | The number of clusters to find.  Tuning this parameter is very important; see notes below. | _(N/A)_ |
| `assignments` | [`arma::Row<size_t>`](../matrices.md) | Vector to store cluster assignments for each point into. | _(N/A)_ |
| `centroids` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix that centroids will be stored into. | _(N/A)_ |
| `initialAssignmentGuess` | `bool` | If `true`, then the values in `assignments` when `Cluster()` is called will be used as the initial clustering. | `false` | | `initialCentroidGuess` | `bool` | If `true`, then the values in `centroids` when `Cluster()` is called will be used as the initial clustering.  Ignored if `initialAssignmentGuess` is also `true`. | `false` |

***Notes***:

 * Selecting the right value of `k` for k-means is very important to ensure
   high-quality results.  If it is not already known how many clusters the data
   contain, heuristics such as the
   [elbow method](https://en.wikipedia.org/wiki/Elbow_method_(clustering)] or
   [other strategies](https://en.wikipedia.org/wiki/K-means_clustering#Optimal_number_of_clusters)
   can be used.

 * Different types can be used for `data` and `centroids` (e.g. `arma::fmat`,
   `arma::sp_mat`, or any matrix type implementing the Armadillo API).  The
   element types of `data` and `centroids` must be the same; `centroids` must be
   the dense equivalent of `data` (e.g. `arma::mat` when `data` is
   `arma::sp_mat`).

 * As an alternative to providing an initial clustering guess, the
   [`InitialPartitionPolicy`](#advanced-functionality-template-parameters)
   allows specification of a different initialization algorithm and mlpack
   offers ready-to-use implementations of a few strategies other than the
   default [`SampleInitialization`](#initialpartitionpolicy).

 * `Cluster()` will terminate early if the cluster distortion drops below `1e-5`
   in a single iteration.
   - Cluster distortion is defined as the square root of the sum of distances
     between each centroids and its value in the previous iteration.

### Other Functionality

 - `km.DistanceComputations()` will return the number of distance computations
   that were performed during the most recent call to `Cluster()` as a `size_t`,
   or `0` if `Cluster()` has not been called yet.

 - `km.Iterations()` will return the number of iterations performed during the
   most recent call to `Cluster()` as a `size_t`, or `0` if `Cluster()` has not
   been called yet.

 - `km.MaxIterations()` returns a `size_t` holding the maximum number of
   iterations to perform during clustering.  `km.MaxIterations() = m` sets the
   maximum number of iterations to `m`.

 - `km.Distance()` returns an instantiated [`DistanceType`](#distancetype) (by
   default, a [`EuclideanDistance`](../core/distances.md#lmetric)).

 - `km.Partitioner()` returns an instantiated
   [`InitialPartitionPolicy`](#initialpartitionpolicy) (by default, a
   [`SampleInitialization`](#initialpartitionpolicy)).

 - `km.EmptyClusterAction()` returns an instantiated
   [`EmptyClusterPolicy`](#emptyclusterpolicy) (by default, a
   [`MaxVarianceNewCluster`](#emptyclusterpolicy)).

 - There is no function to return an instantiated
   [`LloydStepType`](#lloydsteptype), as those are created and destroyed during
   the call to [`Cluster()`](#clustering).

 - A `KMeans` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

### Simple Examples

***Note:*** all of examples in this section use the brute-force iteration
strategy from the original k-means algorithm.  Acceleration can be obtained via
the
[`LloydStepType` template parameter](#advanced-functionality-template-parameters),
detailed in the next section.  See also the
[advanced examples](#advanced-examples).

---

Perform k-means clustering on the satellite dataset and print the average
distance from each point to its assigned centroid.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

// Create KMeans object with default parameters.
mlpack::KMeans km;
arma::mat centroids;
arma::Row<size_t> assignments;
km.Cluster(dataset, 5 /* clusters */, assignments, centroids);

// Compute the average distance from each point to its assigned centroid.
double sumDist = 0.0;
for (size_t i = 0; i < dataset.n_cols; ++i)
{
  sumDist += mlpack::EuclideanDistance::Evaluate(
      dataset.col(i), centroids.col(assignments[i]));
}
const double avgDist = sumDist / (double) dataset.n_cols;

std::cout << "Average distance from a point to its assigned centroid: "
    << avgDist << "." << std::endl;
```

---

Perform k-means clustering on the cloud dataset and print the number of points
assigned to each cluster.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::Load("cloud.csv", dataset, mlpack::Fatal);

// Create KMeans object with default parameters (maximum of 1000 iterations).
mlpack::KMeans km;
arma::Row<size_t> assignments;
km.Cluster(dataset, 4 /* clusters */, assignments);

for (size_t i = 0; i < 4; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
```

---

Perform k-means clustering on the wave energy farm dataset, specifying a maximum
of 500 iterations, and specifying initial centroids.  Print the number of points
assigned to each cluster.

```c++
// See https://datasets.mlpack.org/wave_energy_farm_100.csv.
arma::mat dataset;
mlpack::Load("wave_energy_farm_100.csv", dataset, mlpack::Fatal);

// Create KMeans object with 500 iterations maximum.
mlpack::KMeans km(500);

arma::mat centroids(dataset.n_rows, 6 /* clusters */);
// Pick the first six points as the initial centroids.
for (size_t i = 0; i < 6; ++i)
  centroids.col(i) = dataset.col(i);

// Perform the clustering using the initial centroids as a starting point.
km.Cluster(dataset,
           6 /* clusters */,
           assignments,
           centroids,
           false,
           true /* use centroids as initial guess */);

for (size_t i = 0; i < 6; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
```

---

Perform k-means clustering on the satellite dataset using 32-bit floats to
represent the data, and print the overall SSE (sum-of-squared errors) of the
clustering.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::fmat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

mlpack::KMeans km;
arma::Row<size_t> assignments;
arma::mat centroids;
km.Cluster(dataset, 10 /* clusters */, assignments, centroids);

// Compute the sum-of-squared-errors of the clustering.
double sse = 0.0;
for (size_t i = 0; i < dataset.n_cols; ++i)
{
  sse += mlpack::EuclideanDistance::Evaluate(dataset.col(i),
      centroids.col(assignments[i]));
}
std::cout << "SSE of clustering: " << sse << "." << std::endl;
```

---

Perform k-means clustering on the sparse MovieLens dataset, clustering into a
dense matrix.

```c++
// See https://datasets.mlpack.org/movielens-100k.csv.
arma::sp_mat dataset;
mlpack::Load("movielens-100k.csv", dataset, mlpack::Fatal);

// Create the KMeans object.
mlpack::KMeans km;

// Cluster into 5 clusters.  Note that the centroids are dense!
arma::mat centroids;
km.Cluster(dataset, 5, centroids);

// Print the number of iterations and distance computations during clustering.
std::cout << "Clustering took " << km.Iterations() << " iterations."
    << std::endl;
std::cout << "During clustering, " << km.DistanceComputations() << " were "
    << "computed." << std::endl;
```

### Advanced Functionality: Template Parameters

The `KMeans` class has four template parameters, which allows for extensive
configuration of how the k-means clustering behavior.  The full signature of the
class is:

```
KMeans<DistanceType,
       InitialPartitionPolicy,
       EmptyClusterPolicy,
       LloydStepType>
```

 * [`DistanceType`](#distancetype) is the distance metric to use (default
   [`EuclideanDistance`](../core/distances.md#lmetric)).

 * [`InitialPartitionPolicy`](#initialpartitionpolicy) is the strategy to use to
   assign points to clusters before the first iteration (if no initial guess is
   provided).  The default is `SampleInitialization`, which takes `k` random
   points from the dataset as initial centroids.
   - mlpack also provides `KMeansPlusPlusInitialization`, `RefinedStart`, and
     `RandomPartition`; see the
     [`InitialPartitionPolicy`](#initialpartitionpolicy) documentation for
     details.

 * [`EmptyClusterPolicy`](#emptyclusterpolicy) is the strategy to use when, at
   the end of an iteration, a cluster has no points assigned to it.  The default
   is `MaxVarianceNewCluster`, which finds the point furthest from any centroid
   and sets that to the centroid of the empty cluster.
   - mlpack also provides `KillEmptyClusters` and `AllowEmptyClusters`; see the
     [`EmptyClusterPolicy`](#emptyclusterpolicy) documentation for details.

 * [`LloydStepType`](#lloydsteptype) is the strategy used to actually compute
   the centroids and assignments during the iteration, and this is what should
   be modified to use an accelerated variant of k-means.  The default is
   `NaiveKMeans`, which is the standard algorithmic strategy from the original
   paper.
   - mlpack also provides `ElkanKMeans`, `PellegMooreKMeans`, `HamerlyKMeans`,
     and `DualTreeKMeans`; see the [`LloydStepType`](#lloydsteptype)
     documentation for details.

---

#### `DistanceType`

 * Specifies the distance metric that will be used when clustering.

 * The default distance type is
   [`EuclideanDistance`](../core/distances.md#lmetric).

 * Many [pre-implemented distance metrics](../core/distances.md) are available
   for use, such as [`ManhattanDistance`](../core/distances.md#lmetric) and
   [`ChebyshevDistance`](../core/distances.md#lmetric) and others.

 * [Custom distance metrics](../../developer/distances.md) are easy to
   implement, but *must* satisfy the triangle inequality to provide correct
   results when using an accelerated [`LloydStepType`](#lloydsteptype)
   (e.g. anything other than `NaiveKMeans`), since those accelerations depend on
   the triangle inequality.
   - ***NOTE:*** the cosine distance ***does not*** satisfy the triangle
     inequality.

---

#### `InitialPartitionPolicy`

 * Specifies the strategy to use for initializing clusters, if
   `initialAssignmentGuess` and `initialCentroidGuess` are set to `false` when
   calling [`Cluster()`](#clustering).

 * `SampleInitialization` (the default) selects `k` points randomly from the
   dataset and uses those as initial centroids.
   - This technique is extremely fast and tends to work acceptably in practice.

 * The `KMeansPlusPlusInitialization` class is available for drop-in usage and
   implements the
   [k-means++ algorithm (pdf)](https://courses.cs.duke.edu/spring07/cps296.2/papers/kMeansPlusPlus.pdf).
   - k-means++ uses data points for initial cluster centroids, much like
     `SampleInitialization`, but selects them in a way that prioritizes
     far-apart points.
   - The algorithm takes longer than the trivial `SampleInitialization` but
     tends to provide better clusterings in practice.

 * The `RefinedStart` class is available for drop-in usage and implements the
   [refined start technique (pdf)](https://static.aminer.org/pdf/PDF/000/334/561/refining_initial_points_for_k_means_clustering.pdf).
   - This approach runs k-means several times on small subsamplings of the data,
     and then clusters those results to provide initial seeds.
   - A `RefinedStart` object can be created with the constructor
     `RefinedStart(samplings=100, percentage=0.02)` and passed to the
     [`KMeans` constructor](#constructors).
     * `samplings` represents the number of subsamples of the data to take.
     * `percentage` (between 0 and 1) represents the percentage of data to use
       for each subsample.
   - This algorithm can take significantly longer than either
     `SampleInitialization` or `KMeansPlusPlusInitialization`.
   - In general it will provide better results than `SampleInitialization` but
     it may not provide better results than `KMeansPlusPlusInitialization`.

 * The `RandomPartition` class is available for drop-in usage and assigns each
   point randomly to a cluster, then uses the centroids of those random
   assignments as initial centroids.
   - This approach is fast, but often results in centroids near the overall
     centroid of the data, and this may not lead to good results.

 * A custom `InitialPartitionPolicy` must implement only one of two possible
   functions:

```c++
class CustomInitialPartitionPolicy
{
 public:
  // NOTE: only one of the functions below is required.  The KMeans class will
  // select whichever overload is available (preferring the version that gives
  // centroids, if both are available).

  // Initialize the centroids using the given data and number of clusters.
  //
  //  - `data` is the dataset that `Cluster()` was called with.
  //  - `k` is the number of centroids that `Cluster()` was called with.
  //  - `centroids` should be filled with the initial centroids to use.
  //  - `CentroidsType` will always be a dense matrix type, with the same
  //    element type as `MatType`.
  //
  template<typename MatType, typename CentroidsType>
  void Cluster(const MatType& data,
               const size_t k,
               CentroidsType& centroids);

  // Initialize the point assignments using the given data and number of
  // clusters.
  //
  //  - `data` is the dataset that `Cluster()` was called with.
  //  - `k` is the number of centroids that `Cluster()` was called with.
  //  - `assignments` should be set to size `data.n_cols` and filled with values
  //    between `0` and `k - 1` (inclusive) that represent the cluster
  //    assignment of each point.
  //
  template<typename MatType>
  void Cluster(const MatType& data,
               const size_t k,
               arma::Row<size_t>& assignments);
};
```

---

#### `EmptyClusterPolicy`

 * Specifies the action to make when, at the end of an iteration, a cluster has
   no points assigned to it.

 * `MaxVarianceNewCluster` (the default) will find the point that is furthest
   from the cluster centroid with maximum variance, and assign that point to the
   empty cluster.
   - This is computationally expensive and will perform distance calculations
     between every point and every centroid.
   - However, unless `k` is set very high, the occurrence of an empty cluster at
     the end of an iteration is a rare occurrence.

 * The `KillEmptyClusters` class is available for drop-in usage.
   - This will set a centroid to have all values `DBL_MAX`, and no points will
     be assigned to it in future iterations.
   - If a different `MatType` than `arma::mat` is being used for clustering,
     then the maximum numeric value for that element type will be used instead
     of `DBL_MAX`.
   - Unlike `MaxVarianceNewCluster`, there is effectively no runtime cost for
     `KillEmptyClusters` in the event that an empty cluster is encountered.

 * The `AllowEmptyClusters` class is available for drop-in usage.
   - This leaves a centroid at its previous iteration's value when no points are
     assigned to it.
   - The empty cluster could have points assigned to it in subsequent
     iterations.
   - Unlike `MaxVarianceNewCluster`, there is effectively no runtime cost for
     `AllowEmptyClusters` in the event that an empty cluster is encountered.

 * A custom `EmptyClusterPolicy` must implement only one method that takes the
   [`DistanceType`](#distancetype) and matrix type as template parameters:

```c++
class CustomEmptyClusterPolicy
{
 public:
  // When an empty cluster is encountered, this function will be called, and
  // should make any necessary modifications to `newCentroids`.  If multiple
  // empty clusters are found at the end of an iteration, this function will be
  // called once for every empty cluster that is encountered at the end of an
  // iteration, with different values for `emptyCluster`.
  //
  // - `data`: the dataset that is being clustered.
  // - `emptyCluster`: the index of the cluster that was empty at the end of the
  //     iteration.
  // - `oldCentroids`: the centroids at the beginning of the iteration (i.e.
  //     before the cluster became empty).
  // - `newCentroids`: the centroids at the end of the iteration; any
  //     modifications to the state of the clustering should be made to this
  //     matrix.
  // - `clusterCounts`: number of points assigned to each cluster at the *end*
  //     of the iteration.
  // - `distance`: instantiated DistanceType object to use for distance
  //     computation.
  // - `iteration`: iteration number of the clustering when the empty cluster
  //     was encountered.
  //  - `CentroidsType` will always be a dense matrix type, with the same
  //    element type as `MatType`.
  //
  template<typename DistanceType, typename MatType, typename CentroidsType>
  void EmptyCluster(const MatType& data,
                    const size_t emptyCluster,
                    const CentroidsType& oldCentroids,
                    CentroidsType& newCentroids,
                    arma::Col<size_t>& clusterCounts,
                    DistanceType& distance,
                    const size_t iteration);
}
```

---

#### `LloydStepType`

 * Specifies the strategy to be used during iteration, to recompute the point
   assignments and centroids.

 * `NaiveKMeans` (the default) is the standard k-means algorithm implementation,
   and recomputes assignments by finding the closest centroid of each point with
   brute-force computation, and then recomputes the centroids from those
   assignments.
   - This approach is not accelerated!  At each iteration, it computes
     `data.n_cols` * `k` distances, which can be very slow for large datasets!
   - It is *strongly recommended* to use a different step type, such as one of
     the accelerated variants below, depending on the data.
   - The `NaiveKMeans` strategy is the only strategy mlpack has implemented that
     does *not* rely on the triangle inequality---thus, this is the only
     `LloydStepType` that can be used with a [`DistanceType`](#distancetype)
     that does not satisfy the triangle inequality (such as the cosine
     distance).

 * The `ElkanKMeans` class is available for drop-in usage and implements the
   [accelerated strategy proposed by Charles Elkan (pdf)](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf).
   - This strategy computes how far centroids have moved each iteration, and
     uses the triangle inequality to rule out points that could not possibly
     have changed assignments between iterations.
   - No auxiliary data structures (such as trees) are built for this strategy.
   - This strategy performs well with most datasets, including high-dimensional
     data, although it may not always be the fastest strategy.

 * The `PellegMooreKMeans` class is available for drop-in usage and implements
   a [single-tree kd-tree search strategy for k-means (pdf)](http://reports-archive.adm.cs.cmu.edu/anon/anon/usr/ftp/usr0/ftp/2000/CMU-CS-00-105.pdf).
   - This strategy builds a [`KDTree`](../core/trees/kdtree.md) on the data and
     uses a single-tree traversal much like [`KNN`](knn.md) to find the centroid
     closest to each point in the dataset.
   - The tree only needs to be built at the first iteration; so, the first
     iteration will be slow (because of the tree building) but subsequent
     iterations will be much faster---and continue to get faster as the tree is
     able to prune more.
   - This strategy performs exceedingly well on data in low dimensions (e.g.
     roughly less than 100, but that is just a rule of thumb).

 * The `HamerlyKMeans` class is available for drop-in usage and implements the
   [accelerated strategy proposed by Greg Hamerly (pdf)](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf).
   - This strategy is also based on the triangle inequality and uses
     cluster-to-cluster distances to prune points whose assignments cannot
     change between iterations.
   - Although closely related to `ElkanKMeans`, it is not precisely the same,
     and performance between the two strategies differs depending on the
     dataset.
   - This strategy performs well with most datasets, including high-dimensional
     data, although (like `ElkanKMeans`) it may not always be the fastest
     strategy.

 * The `DualTreeKMeans` class is available for drop-in usage and implements a
   [dual-tree algorithm for k-means (pdf)](https://www.ratml.org/pub/pdf/2017dual.pdf).
   - This strategy builds a [`KDTree`](../core/trees/kdtree.md) on both the
     dataset and the centroids, and much like [`KNN`](knn.md) uses a dual-tree
     algorithm to find the closest centroid for each point in the dataset.
   - The tree on the dataset only needs to be built at the first iteration and
     can be reused; however, the tree on the centroids must be rebuilt at every
     iteration.
   - This strategy benefits from a large value of `k` (e.g. hundreds or
     thousands or more): the cost to build the tree on the centroids must be
     outweighed by the pruning that that tree can do during iteration.
   - This strategy performs best when `k` is very large, and the dataset is
     large and low-dimensional.

 * The `CoverTreeDualTreeKMeans` class is available for drop-in usage and
   implements a [dual-tree algorithm for k-means (pdf)](https://www.ratml.org/pub/pdf/2017dual.pdf).
   - This is the same as `DualTreeKMeans`, except it uses
     [`CoverTree`](../core/trees/cover_tree.md)s instead of
     [`KDTree`](../core/trees/kdtree.md)s.

 * The `DualTreeKMeans` class itself has four template parameters, with the last
   one, `TreeType`, controlling the tree type; this means a custom
   [tree type](../core/trees.md) can be used via a `using` declaration like
   follows:

```c++
template<typename DistanceType, typename MatType, typename CentroidsType>
using OctreeDualTreeKMeans = DualTreeKMeans<DistanceType, MatType,
                                            CentroidsType, Octree>;
```

 * A fully custom `LloydStepType` needs to accept three template parameters and
   implement a constructor and two methods:

```c++
//
// `DistanceType` is the distance metric that is used during iteration (e.g.
// `EuclideanDistance`), `MatType` is the matrix type of the data (e.g.
// `arma::mat`), and `CentroidsType` is the matrix type of the centroids (e.g.
// `arma::mat`).  `CentroidsType` will always be the dense matrix equivalent of
// `MatType`.
//
template<typename DistanceType, typename MatType, typename CentroidsType>
class CustomLloydStepType
{
 public:
  //
  // Construct the CustomLloydStepType object with the given dataset and
  // instantiated distance metric.  This should perform any preprocessing that
  // needs to be done before the first iteration.
  //
  // The dataset and distance are not passed in Iterate(), so it is a good idea
  // to keep a reference to what is passed here.
  //
  CustomLloydStepType(const MatType& dataset, DistanceType& distance);

  //
  // Run a single iteration of k-means, updating the given `centroids` into the
  // `newCentroids` matrix.  If any cluster is empty (that is, if any cluster
  // has no points assigned to it), then the centroid associated with that
  // cluster may be filled with invalid or arbitrary data (it will be corrected
  // later).
  //
  // In addition to updating the centroids, this method should also update
  // `counts` with the number of points assigned to each cluster at the end of
  // the iteration.
  //
  // This function should return the cluster distortion (e.g. the square root of
  // the sum of distances between each centroids and its updated centroid in
  // `newCentroids`).
  //
  double Iterate(const CentroidsType& centroids,
                 CentroidsType& newCentroids,
                 arma::Col<size_t>& counts);

  //
  // Return the number of distance computations performed (so far).
  //
  size_t DistanceCalculations() const;
};
```

---

### Advanced Examples

Perform k-means clustering on the satellite dataset using the Manhattan
distance.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

// Create KMeans object with default parameters and the Manhattan distance.
mlpack::KMeans<mlpack::ManhattanDistance> km;
arma::mat centroids;
arma::Row<size_t> assignments;
km.Cluster(dataset, 5 /* clusters */, assignments, centroids);

// Compute the average distance from each point to its assigned centroid.
double sumDist = 0.0;
for (size_t i = 0; i < dataset.n_cols; ++i)
{
  sumDist += mlpack::ManhattanDistance::Evaluate(
      dataset.col(i), centroids.col(assignments[i]));
}
const double avgDist = sumDist / (double) dataset.n_cols;

std::cout << "Average Manhattan distance from a point to its assigned centroid:"
    << " " << avgDist << "." << std::endl;
```

---

Perform k-means clustering on the wave energy farm dataset, using k-means++ as
the initialization strategy.

```c++
// See https://datasets.mlpack.org/wave_energy_farm_100.csv.
arma::mat dataset;
mlpack::Load("wave_energy_farm_100.csv", dataset, mlpack::Fatal);

// Create KMeans object with k-means++ as the initialization strategy.
mlpack::KMeans<mlpack::EuclideanDistance,
               mlpack::KMeansPlusPlusInitialization> km;

// Perform the clustering using k-means++ for initialization.
arma::mat centroids;
km.Cluster(dataset,
           6 /* clusters */,
           assignments,
           centroids);

for (size_t i = 0; i < 6; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
```

---

Perform k-means clustering on the satellite dataset, killing any empty clusters
at the end of an iteration.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

// Create KMeans object with default parameters, using `KillEmptyClusters` to
// remove any empty clusters when they are encountered.
mlpack::KMeans<mlpack::ManhattanDistance,
               mlpack::SampleInitialization,
               mlpack::KillEmptyClusters> km;
arma::mat centroids;
arma::Row<size_t> assignments;

// Intentionally cluster with very many clusters, so that some will be empty.
km.Cluster(dataset, 500 /* clusters */, assignments, centroids);

// Now compute the number of clusters that are empty.  Since empty clusters have
// their centroids set to DBL_MAX, we only need to look for that.
size_t numEmpty = 0;
for (size_t i = 0; i < centroids.n_cols; ++i)
  if (centroids(0, i) == DBL_MAX)
    ++numEmpty;

std::cout << "After clustering, " << numEmpty << " clusters are empty."
    << std::endl;
```

---

Perform k-means clustering on a subset of the LCDM dataset, using the
Pelleg-Moore tree-based strategy for each iteration.  This provides significant
speedup in low dimensions.

```c++
// See https://datasets.mlpack.org/lcdm_tiny.csv.
arma::mat dataset;
mlpack::Load("lcdm_tiny.csv", dataset);

// Create k-means object with the Pelleg-Moore single-tree strategy for
// clustering.
mlpack::KMeans<mlpack::EuclideanDistance,
               mlpack::SampleInitialization,
               mlpack::MaxVarianceNewCluster,
               mlpack::PellegMooreKMeans> km;

// Perform clustering with 5 clusters.
arma::mat centroids;
arma::Mat<size_t> assignments;
km.Cluster(dataset, 5, assignments, centroids);

// Print statistics about the clustering.
std::cout << "Clustering took " << km.Iterations() << " iterations."
    << std::endl;
std::cout << "During clustering, " << km.DistanceCalculations() << " distance "
    << "calculations were performed." << std::endl;
```

---

Perform k-means clustering on the corel-histogram dataset, using the Elkan
algorithm for acceleration and the Manhattan distance.  This provides
significant speedup in higher dimensions.

```c++
// See https://datasets.mlpack.org/corel-histogram.csv.
arma::mat dataset;
mlpack::Load("corel-histogram.csv", dataset);

// Create k-means object with the Manhattan distance and Elkan's algorithm.
mlpack::KMeans<mlpack::ManhattanDistance,
               mlpack::SampleInitialization,
               mlpack::MaxVarianceNewCluster,
               mlpack::ElkanKMeans> km;

// Perform clustering with 10 clusters.
arma::mat centroids;
km.Cluster(dataset, 10, assignments, centroids);

// Print statistics about the clustering.
std::cout << "Clustering took " << km.Iterations() << " iterations."
    << std::endl;
std::cout << "During clustering, " << km.DistanceCalculations() << " distance "
    << "calculations were performed." << std::endl;
```

---

Perform k-means clustering on the satellite dataset using 32-bit floating point
data, using refined start initialization with custom parameters, allowing empty
clusters to persist at each iteration, and using the Hamerly algorithm for
acceleration.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::fmat dataset;
mlpack::Load("satellite.train.csv", dataset, mlpack::Fatal);

// Create KMeans object with default parameters, using `KillEmptyClusters` to
// remove any empty clusters when they are encountered.
mlpack::KMeans<mlpack::ManhattanDistance,
               mlpack::RefinedStart,
               mlpack::KillEmptyClusters> km;

// Perform clustering with 6 clusters.
arma::fmat centroids;
km.Cluster(dataset, 6, assignments, centroids);

// Print statistics about the clustering.
std::cout << "Clustering took " << km.Iterations() << " iterations."
    << std::endl;
std::cout << "During clustering, " << km.DistanceCalculations() << " distance "
    << "calculations were performed." << std::endl;
```
