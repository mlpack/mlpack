## `EMeans`

The `EMeans` class implements e-means, a mini-batch variant of `k`-means
clustering.  Like `k`-means, it minimizes the sum of squared Euclidean distances
from each point to its closest centroid.  Instead of recomputing every centroid
from the whole dataset at each iteration (as in Lloyd's algorithm), e-means
processes the data in mini-batches.  Each cluster keeps an exponential moving
average of the number and of the sum of the points assigned to it, and its
centroid is the ratio of the two.

The batch size and the step size of the moving average are set by scaling laws
fitted by the authors on large datasets, so the default parameters do not need
to be tuned.  On high-dimensional data with many clusters, e-means typically
reaches the error of 25 Lloyd iterations in 2 to 3 passes over the data, where
each pass costs about as much as one Lloyd iteration.

#### Simple usage example:

```c++
// Use e-means to cluster random data and print the number of points that fall
// into each cluster.

// Create random dataset with two separated 10-dimensional Gaussians.
arma::mat dataset = arma::join_rows(
    arma::randn<arma::mat>(10, 1000) + 3.0,  // 1000 points from N( 3, 1).
    arma::randn<arma::mat>(10, 1000) - 3.0); // 1000 points from N(-3, 1).

mlpack::EMeans em;                              // Step 1: create object.
arma::Row<size_t> assignments;
arma::mat centroids;
em.Cluster(dataset, 2, assignments, centroids); // Step 2: perform clustering.

// Print the number of points in each cluster.
for (size_t c = 0; c < centroids.n_cols; ++c)
{
  std::cout << " * Cluster " << c << " has " << arma::accu(assignments == c)
      << " points." << std::endl;
}
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `EMeans` objects.
 * [`Cluster()`](#clustering): perform clustering.
 * [Other functionality](#other-functionality) for assigning points, inspecting
   and serializing.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   initialization.

#### See also:

 * [mlpack clustering algorithms](../modeling.md#clustering)
 * [k-means on Wikipedia](https://en.wikipedia.org/wiki/K-means_clustering)
 * [e-means: outperforming k-means (in one-tenth the epochs) (arXiv)](https://arxiv.org/abs/2611.09501)

### Constructors

 * `em = EMeans()`
 * `em = EMeans(epochs, batchSize=0, alpha0=0.0, batchLaw=EMeans<>::AUTO, unusedThreshold=-1.0)`
   - Create an `EMeans` object with the given parameters.  The defaults are
     the ones recommended by the authors.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `epochs` | `double` | Number of passes over the data.  A fractional number of epochs runs a partial pass before the last one, which is always a full pass. | `5.0` |
| `batchSize` | `size_t` | Number of points in each mini-batch; `0` sets it with the batch law. | `0` |
| `alpha0` | `double` | Initial step of the moving average, in `(0, 1]`; `0` sets it with the step law. | `0.0` |
| `batchLaw` | `EMeans<>::BatchLaw` | Scaling law that sets the batch size when `batchSize` is `0`: `EMeans<>::THROUGHPUT`, `EMeans<>::QUALITY` (batches about 8 times smaller) or `EMeans<>::AUTO` (`QUALITY` if the data has at most 16 dimensions, `THROUGHPUT` otherwise). | `EMeans<>::AUTO` |
| `unusedThreshold` | `double` | Every 5 batches, clusters whose mass is below this fraction of the mean mass are revived by splitting a cluster chosen in proportion to its mass; `0` disables revival.  A negative value selects `0.1`, or `0` if the data has at most 2 dimensions. | `-1.0` |

***Notes:***

 - The step of the moving average decreases along a cosine schedule from
   `alpha0` to `0.1 * alpha0`.

 - The final centroids are the means of the points assigned to each cluster
   during the last epoch.

 - By default, the initial centroids are points sampled at random from the
   dataset.  This is cheap for many clusters.  For a few clusters,
   [`KMeansPlusPlusInitialization`](#advanced-functionality-template-parameters)
   can avoid poor local minima.

### Clustering

 * `em.Cluster(data, clusters, centroids, initialGuess=false)`
   - Cluster the given `data` into `clusters` clusters, and store the centroids
     in `centroids`.
   - If `initialGuess` is `true`, `centroids` must hold the initial centroids.

---

 * `em.Cluster(data, clusters, assignments, centroids, initialGuess=false)`
   - Cluster the given `data` into `clusters` clusters, store the centroids in
     `centroids`, and store the index of the closest final centroid of each
     point in `assignments`.  This costs one extra pass over the data.
   - If `initialGuess` is `true`, `centroids` must hold the initial centroids.

---

#### Clustering Parameters:

| **name** | **type** | **description** |
|----------|----------|-----------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) dense matrix to cluster. |
| `clusters` | `size_t` | Number of clusters; must be positive and at most the number of points. |
| `assignments` | [`arma::Row<size_t>&`](../matrices.md) | Vector to store the cluster index of each point in. |
| `centroids` | [`arma::mat&`](../matrices.md) | Matrix to store the centroids in (one per column). |
| `initialGuess` | `bool` | Whether `centroids` holds the initial centroids. |

***Notes:***

 - Single-precision data (`arma::fmat`) is supported, in which case
   `centroids` must also be an `arma::fmat`.  Single precision is about twice
   as fast.

### Other functionality

 * `EMeans<>::Assign(data, centroids, assignments)` assigns each point in
   `data` to its closest centroid in `centroids`.  An optional fourth argument
   `&distances` (an `arma::vec`) stores the squared distance of each point to
   its closest centroid.

 * `em.UsedBatchSize()`, `em.UsedAlpha0()`, `em.Steps()` and `em.Splits()`
   return the batch size, the initial step, the number of mini-batch updates
   and the number of revived clusters of the last call to `Cluster()`.

 * `em.Epochs()`, `em.BatchSize()`, `em.Alpha0()`, `em.Law()` and
   `em.UnusedThreshold()` return the parameters, and can be used to modify
   them.

 * An `EMeans` object can be serialized with [`Save()` and
   `Load()`](../load_save.md#mlpack-models-and-objects).  Only the parameters are saved.

### Simple Examples

Cluster single-precision data, compare with the objective of 25 Lloyd
iterations, and assign new points.

```c++
// Random data: 20000 points in 32 dimensions.
arma::fmat data(32, 20000, arma::fill::randn);

// Cluster into 100 clusters with 3 epochs, from the same initial centroids
// as k-means.
arma::fmat init = data.cols(arma::randperm(data.n_cols, 100));
arma::fmat centroids = init;
mlpack::EMeans em(3.0);
em.Cluster(data, 100, centroids, true);

arma::Row<size_t> assignments;
arma::fvec distances;
mlpack::EMeans<>::Assign(data, centroids, assignments, &distances);
std::cout << "e-means objective after 3 epochs: " << arma::accu(distances)
    << " (" << em.Steps() << " mini-batches of " << em.UsedBatchSize()
    << " points)." << std::endl;

// The same with 25 iterations of k-means.
arma::mat kmCentroids = arma::conv_to<arma::mat>::from(init);
mlpack::KMeans km(25);
km.Cluster(arma::conv_to<arma::mat>::from(data), 100, kmCentroids, true);
arma::vec kmDistances;
mlpack::EMeans<>::Assign(arma::conv_to<arma::mat>::from(data), kmCentroids,
    assignments, &kmDistances);
std::cout << "k-means objective after 25 iterations: "
    << arma::accu(kmDistances) << "." << std::endl;

// Assign new points to their closest e-means centroid.
arma::fmat newPoints(32, 10, arma::fill::randn);
mlpack::EMeans<>::Assign(newPoints, centroids, assignments);
std::cout << "Clusters of the new points: " << assignments;
```

### Advanced Functionality: Template Parameters

The `EMeans` class has one template parameter:

```
EMeans<InitialPartitionPolicy>
```

 * `InitialPartitionPolicy` selects the initial centroids when
   `initialGuess` is `false`.  It defaults to `SampleInitialization`, which
   samples random points of the dataset.  The policies of
   [`KMeans`](../modeling.md#clustering) that give centroids, such as
   `KMeansPlusPlusInitialization`, can be used; `KMeansPlusPlusInitialization`
   supports only `arma::mat` data.  A custom policy must implement one
   function:

```c++
class CustomInitialization
{
 public:
  // Store `clusters` initial centroids for `data` in `centroids`, one per
  // column.  `CentroidsType` is `arma::mat` or `arma::fmat`, matching `data`.
  template<typename MatType, typename CentroidsType>
  void Cluster(const MatType& data,
               const size_t clusters,
               CentroidsType& centroids);
};
```

```c++
// Use k-means++ to initialize e-means.
arma::mat data(10, 5000, arma::fill::randn);
arma::mat centroids;
mlpack::EMeans<mlpack::KMeansPlusPlusInitialization> em;
em.Cluster(data, 8, centroids);
```
