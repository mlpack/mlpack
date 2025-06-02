## `MeanShift`

The `MeanShift` class implements mean shift, a clustering technique.  Mean shift
models the density of the data using a kernel function (also called Parzen
window), producing a number of clusters that represent the data density.  Mean
shift does not require the user to guess the number of clusters, and does not
make any assumptions on the shape of the data.

mlpack's `MeanShift` class allows control of the kernel function used via
template parameters.

#### Simple usage example:

```c++
// Use mean shift to cluster random data and print the number of points that
// fall into each cluster.

// Create random dataset with two separated 10-dimensional Gaussians.
arma::mat dataset = arma::join_rows(
    arma::randn<arma::mat>(10, 1000) + 3.0,  // 1000 points from N(-3, 1).
    arma::randn<arma::mat>(10, 1000) - 3.0); // 1000 points from N( 3, 1).

mlpack::MeanShift ms;                        // Step 1: create object.
arma::Row<size_t> assignments;
arma::mat centroids;
ms.Cluster(dataset, assignments, centroids); // Step 2: perform clustering.

// Print the number of clusters.
std::cout << "Found " << centroids.n_cols << " centroids." << std::endl;

// Print the number of points in each cluster.
for (size_t c = 0; c < centroids.n_cols; ++c)
{
  std::cout << " * Cluster " << c << " has " << arma::accu(assignments == c)
      << " points." << std::endl;
}
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `MeanShift` objects.
 * [`Cluster()`](#clustering): perform clustering.
 * [Other functionality](#other-functionality) for loading, saving, inspecting,
   and estimating the radius to use.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [mlpack clustering algorithms](../modeling.md#clustering)
 * [mlpack kernels](../core/kernels.md)
 * [Mean shift on Wikipedia](https://en.wikipedia.org/wiki/Mean_shift)
 * [Mean Shift, Mode Seeking, and Clustering (pdf)](http://users.isr.ist.utl.pt/~alex/Resources/meanshift.pdf)

### Constructors

 * `ms = MeanShift(radius=0, maxIterations=1000)`
   - Create a `MeanShift` object that will use the
     default [`GaussianKernel`](../core/kernels.md#gaussiankernel) to weight
     points for cluster centroid recalculations.

---

 * `ms = MeanShift<false>(radius=0, maxIterations=1000)`
   - Create a `MeanShift` object that will not weight points differently when
     recalculating cluster centroids.
   - Centroid recalculation will use all points within a distance of `radius`
     from the current cluster centroid, uniformly weighted.

---

 * `ms = MeanShift(radius, maxIterations, kernel)`
   - Create a `MeanShift` object that will use the given `kernel` object (a
     `GaussianKernel`) for weighting points during cluster centroid
     recalculations.

---

 * `ms = MeanShift<true, KernelType>(radius, maxIterations, kernel=KernelType())`
   - Create a `MeanShift` object that will use the given
     [`KernelType`](../core/kernels.md) for weighting points during cluster
     centroid recalculations.
   - [mlpack kernels](../core/kernels.md) or custom kernel classes implementing
     a [`Gradient()` function](#advanced-functionality-template-parameters) can
     be used for the `KernelType` template parameter.
   - If `kernel` is not specified, a default-constructed `KernelType` will be
     used.
   - A list of usable `KernelType`s supplied with mlpack can be found in the
     [advanced functionality section](#advanced-functionality-template-parameters).

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `radius` | `double` | Radius around each centroid for weighting during centroid recomputation.  Larger means higher weights for faraway points.  Values less than or equal to 0 mean that the radius will be estimated from data. | `0.0` |
| `maxIterations` | `size_t` | Maximum number of iterations of the mean shift algorithm to run. | `1000` |
| `kernel` | [`KernelType`](#advanced-functionality-template-parameters) | Instantiated kernel object to use for density calculations. | [`GaussianKernel()`](../core/kernels.md#gaussiankernel) |

***Notes:***

 - A larger `radius` value will generally result in fewer clusters (e.g. a
   coarser clustering); smaller `radius` values will generally result in more
   clusters.

 - When `MeanShift<false>` is used, `radius` is the hard distance threshold for
   points to be considered in the recomputation of a centroid.

### Clustering

 * `ms.Cluster(data, centroids, forceConvergence=true, useSeeds=true)`
   - Cluster the given data, storing the resulting cluster centroids in
     `centroids`.
   - `centroids` will be set to size `data.n_rows` x `numClusters`, where
     `numClusters` is the number of clusters found by the mean shift algorithm.
   - The `i`th cluster centroid can be obtained with `clusters.col(i)`.

---

 * `ms.Cluster(data, assignments, centroids, forceConvergence=true, useSeeds=true)`
   - Cluster the given data, storing the resulting cluster centroids in
     `centroids` and cluster assignments for each data point in `assignments`.
   - `centroids` will be set to size `data.n_rows` x `numClusters`, where
     `numClusters` is the number of clusters found by the mean shift algorithm.
   - `assignments` will be set to length `data.n_cols`; the assignment of the
     `i`th point can be obtained with `assignments[i]`.
   - The cluster centroid of the `i`th point's cluster can be obtained with
     `centroids.col(assignments[i])`.

---

#### Clustering Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `data` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix holding the dataset to be clustered. | _(N/A)_ |
| `centroids` | [`arma::mat`](../matrices.md) | [Column-major](../matrices.md#representing-data-in-mlpack) matrix that centroids will be stored into. | _(N/A)_ |
| `assignments` | [`arma::Row<size_t>`](../matrices.md) | Vector to store cluster assignments for each point into. | _(N/A)_ |
| `forceConvergence` | `bool` | If `true`, forces convergence of every cluster, ignoring `maxIterations`. | `false` |
| `useSeeds` | `bool` | If `true`, estimates of high-density regions in the dataset will be used as initial centroids, instead of the full dataset. | `true`

***Notes***:

 * It is recommended to leave `useSeeds` to its default value of `true`.  When
   `useSeeds` is set to `false`, the entire dataset is used as the initial set
   of centroids.  For large datasets, this can be slow!

 * Different types can be used for `data` and `centroids` (e.g., `arma::fmat` or
   any dense matrix type implementing the Armadillo API).  The types of `data`
   and `centroids` must be the same.

### Other Functionality

 * A `MeanShift` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `EstimateRadius(data, ratio=0.2)` returns a `double` that estimates a good
   value to use for the radius parameter.  `ratio` (between 0 and 1) controls
   the percentage of the dataset used for the estimate.
   - This function is called internally by `Cluster()` at the start of
     clustering to choose a radius, if `radius` is less than or equal to 0.

 * As an alternative to constructor parameters, the radius can be set with
   `ms.Radius(newRadius)`, and the maximum number of iterations can be set with
   `ms.MaxIterations() = newMaxIter`.

 * `ms.Radius()` returns the current radius for clustering.
   `ms.Radius(r)` sets the radius to `r`.

 * `ms.MaxIterations()` returns the current maximum number of iterations for
   clustering.  `ms.MaxIterations() = m` sets the maximum number of iterations
   to `m`.

### Simple Examples

Perform mean shift clustering on the satellite dataset and print the average
distance from each point to its assigned centroid.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat dataset;
mlpack::data::Load("satellite.train.csv", dataset, true);

// Create MeanShift object with default parameters and perform clustering.
mlpack::MeanShift ms;
arma::mat centroids;
arma::Row<size_t> assignments;
ms.Cluster(dataset, assignments, centroids);

// Print the number of clusters.
std::cout << "MeanShift computed " << centroids.n_cols << " clusters."
    << std::endl;

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

Perform mean shift clustering with custom settings of `radius` and
`maxIterations` on the wave energy farm dataset, using `EstimateRadius()`
to set the initial radius.

```c++
// See https://datasets.mlpack.org/wave_energy_farm_100.csv.
arma::mat dataset;
mlpack::data::Load("wave_energy_farm_100.csv", dataset, true);

// Create MeanShift object and set parameters.
mlpack::MeanShift ms;
const double radiusEstimate = ms.EstimateRadius(dataset, 0.2);

// Use 2x the estimate for a coarser clustering.
ms.Radius(2.0 * radiusEstimate);
// Use only 100 iterations.
ms.MaxIterations() = 100;

// Perform the clustering.
arma::mat centroids;
ms.Cluster(dataset, centroids);

std::cout << "MeanShift found " << centroids.n_cols << " clusters."
    << std::endl;

// Save the centroids to disk.
mlpack::data::Save("wave_energy_centroids.csv", centroids);
```

---

Perform mean shift clustering with no kernel (e.g. unit weighting of points in a
centroid) on the cloud dataset.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::mat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Don't use a kernel for clustering.  This means all points within the radius
// are weighted equally.  Use a custom radius of 25.
mlpack::MeanShift<false> ms(25.0, 100 /* max iterations */);

arma::mat centroids;
arma::Row<size_t> assignments;
ms.Cluster(dataset, assignments, centroids);

// Print the number of clusters and the number of points in each cluster.
std::cout << "MeanShift found " << centroids.n_cols << " clusters."
    << std::endl;
for (size_t i = 0; i < centroids.n_cols; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
```

---

Perform mean shift clustering with the triangular kernel on the cloud dataset,
using 32-bit floating point matrices to represent the data.

```c++
// See https://datasets.mlpack.org/cloud.csv.
arma::fmat dataset;
mlpack::data::Load("cloud.csv", dataset, true);

// Create the MeanShift object using a TriangularKernel.
mlpack::TriangularKernel tk;
mlpack::MeanShift<true, mlpack::TriangularKernel> ms(50.0 /* radius */,
                                                     1000 /* max iterations */,
                                                     tk);

// Perform clustering.
arma::fmat centroids;
arma::Row<size_t> assignments;
ms.Cluster(dataset, assignments, centroids);

// Print the number of clusters and the number of points in each cluster.
std::cout << "MeanShift found " << centroids.n_cols << " clusters."
    << std::endl;
for (size_t i = 0; i < centroids.n_cols; ++i)
{
  std::cout << " - Cluster " << i << " has " << arma::accu(assignments == i)
      << " points assigned to it." << std::endl;
}
```

---

### Advanced Functionality: Template Parameters

The `MeanShift` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```
MeanShift<UseKernel, KernelType>
```

 * `UseKernel` (default `true`) is a `bool` parameter representing whether a
   kernel function is used to weight points during centroid recomputation.  If
   it is `false`, then each point within distance `radius` of the centroid will
   be used (without weighting) to recompute the centroid.  This strategy (with
   `UseKernel = false`) is also known as using a 'flat kernel'.

 * `KernelType` represents the kernel function (or Parzen window) to be used to
   weight points during centroid recomputation.  Although many
   [mlpack kernels](../core/kernels.md) are available, only those with
   `Gradient()` functions (described below) are supported.  Available kernels
   for drop-in usage include:
   - [`GaussianKernel`](../core/kernels.md#gaussiankernel) *(default)*
   - [`EpanechnikovKernel`](../core/kernels.md#epanechnikovkernel)
   - [`LaplacianKernel`](../core/kernels.md#laplaciankernel)
   - [`SphericalKernel`](../core/kernels.md#sphericalkernel) *(note: this is
     equivalent to the flat kernel, or, setting `UseKernel = false`)*
   - [`TriangularKernel`](../core/kernels.md#triangularkernel)

Custom kernels for mean shift can be easily implemented, and must implement only
one function (`Gradient()`):

```c++
class CustomKernel
{
  // Evaluate the gradient of the kernel function given the distance between two
  // points.  Specifically, given that the kernel function is K(t) (where t is
  // the distance between the two points), this function should return K'(t).
  double Gradient(const double t);
};
```
