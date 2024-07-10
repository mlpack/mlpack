## `MeanShift`

The `MeanShift` class implements mean shift, a clustering technique.  Mean shift
models the density of the data using a specified kernel function, producing a
number of clusters that represent the data density.  Mean shift does not require
the user to guess the number of clusters, and does not make any assumptions on
the shape of the data.

mlpack's `MeanShift` class allows control of the kernel function used via
template parameters.

#### Simple usage example:

```c++
// Use mean shift to cluster random data and print the number of points that
// fall into each cluster.

// All data is uniform random 10-dimensional; replace with a data::Load() call
// or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu);

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
 * [`Cluster()`](#cluster): perform clustering.
 * [Other functionality](#other-functionality) for loading, saving, inspecting,
   and estimating the radius to use.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-template-parameters) for custom
   behavior.

#### See also:

 * [mlpack clustering algorithms](../../index.md#clustering-algorithms)
 * [mlpack kernels](../core.md#kernels)
 * [Mean shift on Wikipedia](https://en.wikipedia.org/wiki/Mean_shift)
 * [Mean Shift, Mode Seeking, and Clustering (pdf)](http://users.isr.ist.utl.pt/~alex/Resources/meanshift.pdf)

### Constructors

 * `ms = MeanShift(radius=0, maxIterations=1000)`

---

 * `ms = MeanShift<false>(radius=0, maxIterations=1000)`

---

 * `ms = MeanShift(radius, maxIterations, kernel)`

---

 * `ms = MeanShift<true, KernelType>(radius, maxIterations, kernel)`

---

### Clustering

 * `ms.Cluster(data, centroids, forceConvergence=true, useSeeds=true)`

---

 * `ms.Cluster(data, assignments, centroids, forceConvergence=true, useSeeds=true)`

---

### Other Functionality

 * A `MeanShift` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

 * `EstimateRadius()`

### Simple Examples

Perform mean shift clustering on the satellite dataset and print the average
distance from each point to its assigned centroid.

```c++

```

---

Perform mean shift clustering with custom settings of `radius` and
`maxIterations` on a subset of the covertype dataset, using `EstimateRadius()`
to set the initial radius.

```c++
```

---

Perform mean shift clustering with no kernel (e.g. unit weighting of points in a
centroid) on the cloud dataset.

```c++

```

---

Perform mean shift clustering with the triangular kernel on the cloud dataset,
using 32-bit floating point matrices to represent the data.

```c++

```

---

Perform mean shift clustering on a random sparse dataset.

```c++

```

---

### Advanced Functionality: Template Parameters
