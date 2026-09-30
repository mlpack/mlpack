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

 * `km.Cluster(data, k, centroids, initialCentroidGuess=false)`
   - Cluster the given data into `k` clusters, storing computed centroids into
     the `centroids` matrix.
   - `centroids` will be set to size `data.n_rows` x `k`; the `i`th cluster
     centroid can be obtained with `centroids.col(i)`.
   - If `initialCentroidGuess` is `true`, then `centroids` is expected to have
     size `data.n_rows` x `k` when `Cluster()` is called, and those centroids
     are used as the initial clustering.

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

 * Different types can be used for `data` and `centroids` (e.g. `arma::fmat` or
   any dense matrix type implementing the Armadillo API).  The types of `data`
   and `centroids` must be the same.

 * As an alternative to providing an initial clustering guess, the
   [`InitialPartitionPolicy`](#advanced-functionality-template-parameters)
   allows specification of a different initialization algorithm and mlpack
   offers ready-to-use implementations of a few strategies other than the
   default [`SampleInitialization`](#initialpartitionpolicy).

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

KMeans km;
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
   provided).  The default is `SampleInitialization`, which takes 

 * [`EmptyClusterPolicy`](#emptyclusterpolicy) is the strategy to use when, at
   the end of an iteration, a cluster has no points assigned to it.  The default
   is `MaxVarianceNewCluster`, which 

 * [`LloydStepType`](#lloydsteptype) is the strategy used to actually compute
   the centroids and assignments during the iteration, and this is what should
   be modified to use an accelerated variant of k-means.  The default is
   `NaiveKMeans`, which is the standard algorithmic strategy from the original
   paper.

---

#### `DistanceType`

---

#### `InitialPartitionPolicy`

---

#### `EmptyClusterPolicy`

---

#### `LloydStepType`

---

### Advanced Examples

Perform k-means clustering on the satellite dataset using the Manhattan
distance.

```c++

```

---

Perform k-means clustering on the wave energy farm dataset, using k-means++ as
the initialization strategy.

```c++

```

---

Perform k-means clustering on the satellite dataset, killing any empty clusters
at the end of an iteration.

```c++

```

---

Perform k-means clustering on a subset of the LCDM dataset, using the
Pelleg-Moore tree-based strategy for each iteration.  This provides significant
speedup in low dimensions.

```c++

```

---

Perform k-means clustering on the corel-histogram dataset, using the Elkan
algorithm for acceleration and the Manhattan distance.  This provides
significant speedup in higher dimensions.

```c++

```

---

Perform k-means clustering on the satellite dataset using 32-bit floating point
data, using refined start initialization, allowing empty clusters to persist at
each iteration, and using the Hamerly algorithm for acceleration.

```c++

```
