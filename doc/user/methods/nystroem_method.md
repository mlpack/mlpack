## `NystroemMethod`: kernel matrix approximation

The `NystroemMethod` class implements the Nyström method, a technique useful for
kernel-based algorithms on large-scale data.  Almost all kernel-based algorithms
(like [`KernelPCA`](kernel_pca.md)) depend on the computation of the kernel
matrix `K`, which is a large square matrix whose dimensions are the same as the
number of points in the data---so, for datasets above ~100k points, it is often
infeasible to compute `K` at all.

The Nyström method approximates the kernel matrix `K` as a low-rank matrix,
using a small sample of the points in the dataset.  It computes the low-rank
matrix `G`, such that

```
K ~= G * G.t()
```

mlpack's `NystroemMethod` class supports arbitrary kernel function types
and arbitrary strategies for selecting the sample of points to use.  The class
is most useful as a building block for machine learning techniques; it is not a
standalone classifier or regressor.

#### Simple usage example:

```c++
// Data is uniform random and 10-dimensional.  Replace with a Load() call
// or similar for a real application.
arma::mat dataset(10, 10000, arma::fill::randu); // 10000 points.

// Step 1: create object.
mlpack::NystroemMethod<mlpack::GaussianKernel> n(500 /* samples */);

// Step 2: compute approximation.
arma::mat g;
n.Apply(dataset, g);

// Step 3: optionally reconstruct full approximated kernel matrix.
arma::mat approxK = g * g.t();

std::cout << "Minimum approximated kernel value: "
    << arma::min(arma::min(approxK)) << "." << std::endl;
std::cout << "Maximum approximated kernel value: "
    << arma::max(arma::max(approxK)) << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): construct `NystroemMethod` objects.
 * [`Apply()`](#applying-transformations): apply the Nyström method to
   approximate a kernel matrix.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-template-parameters) for
   different sampling strategies.
 * [Advanced examples](#advanced-examples) that make use of different sampling
   strategies via template parameters.

### See also:

 * [`KernelPCA`](kernel_pca.md): kernel principal components analysis
 * [mlpack kernels](../core/kernels.md): list of kernels implemented by mlpack
 * [mlpack transformations](../transformations.md)
 * [Kernel method on Wikipedia](https://en.wikipedia.org/wiki/Kernel_method)
 * [Low-rank matrix approximations on Wikipedia](https://en.wikipedia.org/wiki/Low-rank_matrix_approximation)
 * [Using the Nyström method to speed up kernel machines (pdf)](https://proceedings.neurips.cc/paper_files/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf)

### Constructors

 * `n = NystroemMethod<KernelType>(samples)`
 * `n = NystroemMethod<KernelType>(samples, kernel)`
   - Create a `NystroemMethod` object that will use `samples` points to
     approximate the kernel matrix of data.
   - `KernelType` is the type of kernel to be used.  This can be any of the
     pre-implemented [mlpack kernels](../core/kernels.md) or a
     [custom kernel](../core/kernels.md#implement-a-custom-kernel).
   - When approximating the kernel matrix, points will be computed as the
     centroids of 5 iterations of [`KMeans`](kmeans.md).
   - Optionally, pass an instantiated `KernelType` to use as the kernel during
     approximation.

 * `n = NystroemMethod<KernelType, PointSelectionPolicy>(samples)`
 * `n = NystroemMethod<KernelType, PointSelectionPolicy>(samples, kernel)`
 * `n = NystroemMethod<KernelType, PointSelectionPolicy>(samples, kernel, pointSelector)`
   - Create a `NystroemMethod` object that will use `samples` points to
     approximate the kernel matrix of data.
   - `KernelType` is the type of kernel to be used.  This can be any of the
     pre-implemented [mlpack kernels](../core/kernels.md) or a
     [custom kernel](../core/kernels.md#implement-a-custom-kernel).
   - `PointSelectionPolicy` is the strategy to use to select points from the
     dataset.  See the
     [advanced functionality section](#advanced-functionality-template-parameters)
     for more details.
   - Optionally, pass an instantiated `KernelType` to use as the kernel during
     approximation.

 * `n = NystroemMethod<KernelType>()`
   - Create a `NystroemMethod` object that will use 100 points to approximate
     the kernel matrix of data.
   - `KernelType` is the type of kernel to be used.  This can be any of the
     pre-implemented [mlpack kernels](../core/kernels.md) or a
     [custom kernel](../core/kernels.md#implement-a-custom-kernel).
   - 100 is not generally the right number of samples to use for an
     approximation!  It is strongly suggested that the number of samples be set
     with [`n.Samples(s)`](#other-functionality) or the constructors above
     should be used.

#### Constructor parameters

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `KernelType` | [kernel class](../core/kernels.md) | Template parameter specifying which kernel to use for the Nyström method. | _(N/A)_ |
| `PointSelectionPolicy` | [point selection policy class](#advanced-functionality-template-parameters) | Template parameter specifying the strategy to use to sample points for kernel matrix approximation. | [`KMeansSelection<>`](#kmeansselection) |
|----------|----------|-----------------|-------------|
| `samples` | `size_t` | Number of points to use in kernel matrix approximation. | `100` |
| `kernel` | [`KernelType`](#advanced-functionality-template-parameters) | Optional instantiated kernel to use for computing kernel values. | `KernelType()` |
| `pointSelector` | [`PointSelectionPolicy`](#advanced-functionality-template-parameters) | Optional instantiated object to use for selecting points to use as samples for the kernel matrix approximation. | `PointSelectionPolicy()` |

### Applying transformations

Once a `NystroemMethod` object has been created, it can be used to approximate
the kernel matrix of any data matrix.

 * `n.Apply(data, g)`
   - Approximate the kernel matrix of `data` as a low-rank matrix, storing the
     decomposed approximation into the matrix `g`.
   - Both `data` and `g` should have the same type `arma::mat`, or any other
     dense matrix type satisfying the Armadillo API.
   - The matrix `g` will be set to size `data.n_cols` x `samples`.
   - The approximate kernel matrix can be computed as `g * g.t()`.

***Note:*** the quality of the approximation depends heavily on the number of
samples used (i.e. the rank of the decomposition).  More samples means a better
approximation, but will take longer to compute.

### Other functionality

 * A `NystroemMethod` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

 * `n.Samples()` will return a `size_t` indicating how many samples will be used
   for approximation.  `n.Samples(newSamples)` will set the number of samples to
   use to `newSamples`.

 * `n.Kernel()` will return a
   [`KernelType&`](#advanced-functionality-template-parameters) that is the
   instantiated kernel object to use for kernel value computations.

 * `n.PointSelector()` will return a
   [`PointSelectionPolicy&`](#advanced-functionality-template-parameters) that
   is the instantiated object to use to select samples from the data for
   approximating the kernel matrix.

### Simple examples

Use the Nyström method to approximate the kernel matrix of the cloud dataset
when using a polynomial kernel of degree 3.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", data, mlpack::Fatal);

// Instantiate the kernel with a degree of 3.
mlpack::PolynomialKernel pk(3);
mlpack::NystroemMethod<mlpack::PolynomialKernel> n(50 /* samples */, pk);

// Approximate the kernel matrix.
arma::mat g;
n.Apply(data, g);

std::cout << "Size of low-rank approximation: " << g.n_rows << " x " << g.n_cols
    << "." << std::endl;
```

---

Use the Nyström method to approximate the kernel matrix of the circles dataset
with different numbers of samples, and compare the reconstruction error of each
against the exact kernel matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);

// First, compute the exact kernel matrix.
mlpack::GaussianKernel gk;
arma::mat trueKernel(data.n_cols, data.n_cols);
for (size_t i = 0; i < data.n_cols; ++i)
  for (size_t j = i; j < data.n_cols; ++j)
    trueKernel(j, i) = gk.Evaluate(data.col(i), data.col(j));
// Reflect the lower triangle to the upper triangle.
trueKernel = arma::symmatl(trueKernel);

// Now try several different numbers of samples and compute the exact
// reconstruction error.
mlpack::NystroemMethod<mlpack::GaussianKernel> n(10, gk);
for (size_t samples = 10; samples <= 100; samples += 10)
{
  n.Samples(samples);
  arma::mat g;
  n.Apply(data, g);

  // Construct the full approximate kernel matrix.
  arma::mat approxKernel = g * g.t();

  // Print the reconstruction error.
  std::cout << "With " << samples << " samples, L2 norm of reconstruction error"
      << " is " << arma::norm(trueKernel - approxKernel, 2) << "." << std::endl;
}
```

---

Create a `NystroemMethod` object and save it to disk for later usage.

```c++
mlpack::NystroemMethod<mlpack::LaplacianKernel> n(
    100, /* samples */
    mlpack::LaplacianKernel(3.0));

// Save the object to `nm.bin`.
mlpack::Save("nm.bin", n, mlpack::Fatal);
```

---

Load a `NystroemMethod` object from disk and print information about the
kernel, then approximate the kernel matrix of the cloud dataset using 32-bit
floating point values.

```c++
// Load from `nm.bin`.
mlpack::NystroemMethod<mlpack::LaplacianKernel> n;
mlpack::Load("nm.bin", n, mlpack::Fatal);

// Now load the cloud dataset into an arma::fmat and approximate the kernel
// matrix.
arma::fmat dataset;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", dataset, mlpack::Fatal);

arma::fmat g;
n.Apply(dataset, g);

// Compute the singular values of the approximation.
arma::fvec s = arma::svd(g);
std::cout << "Top 10 singular values of low-rank approximate kernel matrix: "
    << s.subvec(0, 9).t();
```

### Advanced functionality: template parameters

The `NystroemMethod` class has two template parameters for configurable
behavior.  The full signature of the class is:

```
NystroemMethod<KernelType, PointSelectionPolicy>
```

 * `KernelType` is the type of kernel function to use when approximating the
   kernel matrix.  There are many [mlpack kernels](../core/kernels.md) available
   for drop-in usage, and
   [implementing a custom kernel](../core/kernels.md#implement-a-custom-kernel)
   is quite straightforward.  Common choices of kernels include:
   - [`GaussianKernel`](../core/kernels.md#gaussiankernel)
   - [`CauchyKernel`](../core/kernels.md#cauchykernel)
   - [`LaplacianKernel`](../core/kernels.md#laplaciankernel)
   - [`PolynomialKernel`](../core/kernels.md#polynomialkernel)
   - [`SphericalKernel`](../core/kernels.md#sphericalkernel)

 * `PointSelectionPolicy` specifies the strategy to use to sample points from
   the dataset to construct the low-rank kernel matrix approximation.
   - [`KMeansSelection<>`](#kmeansselection) (the default) uses a few iterations
     of [`KMeans`](kmeans.md) to compute `samples` clusters, and uses the
     centroids of those clusters as points for approximating the kernel matrix.
   - [`RandomSelection<>`](#randomselection) is available for drop-in usage and
     will select `samples` points randomly from the dataset.
   - [`OrderedSelection<>`](#orderedselection) is available for drop-in usasge
     and will select the first `samples` points from the dataset.
   - [Custom `PointSelectionPolicy` classes](#custom-pointselectionpolicy-classes)
     are straightforward to implement and be used for fully custom behavior.

#### `KMeansSelection`

The `KMeansSelection<>` class implements a strategy for point sampling that
performs [`KMeans`](kmeans.md) clustering on the dataset, and uses the centroids
as points for kernel matrix approximation.

 * `KMeansSelection` (with no template parameters) will use the standard
   [`KMeans`](kmeans.md) class and perform 5 iterations of k-means clustering.

 * `KMeansSelection<KMeansType, iterations>` will use the given `KMeansType` for
   clustering and perform `iterations` iterations to find the centroids to use.
   - `KMeansType` must be the [`KMeans`](kmeans.md) class, with custom
     [template parameters](kmeans.md#advanced-functionality-template-parameters)
     specified.
   - `iterations` must be a `size_t`.  If `0`, k-means will run until
     convergence *(not recommended!)*.  Generally a small iteration count is
     sufficient for a decent approximation.

#### `RandomSelection`

The `RandomSelection` class selects points randomly from the dataset.  It has no
configurable parameters.  Although `RandomSelection` is more efficient than
`KMeansSelection`, it will likely produce worse approximations.

#### `OrderedSelection`

The `OrderedSelection` class selects the first `samples` points from the
dataset.  It has no configurable parameters.  `OrderedSelection` is only likely
to perform well if the dataset is either already shuffled, or it is known that
the first `samples` points are representative of the dataset.

#### Custom `PointSelectionPolicy` classes

Implementing a custom `PointSelectionPolicy` is straightforward and requires two
or three functions to be implemented:

```c++
class CustomPointSelectionPolicy
{
  // A default constructor must be available.  (Optionally, more constructors
  // that take parameters can be added.)
  CustomPointSelectionPolicy();

  //
  // Only *one* of the two overloads below is required.  If you intend to
  // generate new points for approximation (like `KMeansSelection`), use the
  // version that returns a `MatType`.  If you intend to sample points (like
  // `RandomSelection`), use the version that returns an `arma::Col<size_t>`.
  //

  // Given the dataset `data`, and a desired number of points `m`, return a
  // vector of indices of length `m` that specifies which points in `data`
  // should be used to approximate the kernel matrix.
  //
  // `MatType` will be a dense matrix type that follows the Armadillo API (e.g.
  // arma::mat, arma::fmat, etc.).
  template<typename MatType>
  arma::Col<size_t> Select(const MatType& data, const size_t m);

  // Given the dataset `data`, and a desired number of points `m`, return a
  // matrix with `m` points that will be used to approximate the kernel matrix.
  // The returned matrix should have size `data.n_rows` x `m`.
  //
  // `MatType` will be a dense matrix type that follows the Armadillo API (e.g.
  // arma::mat, arma::fmat, etc.).
  template<typename MatType>
  MatType Select(const MatType& data, const size_t m);

  // Serialize the object with cereal.  For more information, see the cereal
  // documentation or look at the implementation of other mlpack classes to get
  // an idea.
  //
  // This method is not needed if you never plan to call mlpack::Load() or
  // mlpack::Save() for serialization.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version);
};
```

### Advanced examples

Use the Nyström method with random point selection to approximate the kernel
matrix of the cloud dataset with the Epanechnikov kernel.

```c++
arma::mat dataset;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", dataset, mlpack::Fatal);

mlpack::NystroemMethod<mlpack::EpanechnikovKernel,
                       mlpack::RandomSelection> n(50);
arma::mat g;
n.Apply(dataset, g);

// Print the first 10 singular values of the low-rank approximation.
arma::vec s = arma::svd(g);
std::cout << "First ten singular values of low-rank kernel matrix "
    << "approximation: " << s.subvec(0, 9).t();
```

---

Use the Nyström method with k-means point selection to approximate the kernel
matrix of the corel dataset with the Gaussian kernel, and use the Hamerly
algorithm for efficient k-means clustering.

```c++
arma::mat dataset;
mlpack::Load("http://datasets.mlpack.org/corel-histogram.csv", dataset,
    mlpack::Fatal);

// We will use a custom k-means type so that we can control the behavior of the
// clustering.  The Hamerly algorithm is generally more efficient than the
// default.  We will also remove any clusters that contain no points.
mlpack::NystroemMethod<mlpack::GaussianKernel,
                       mlpack::KMeansSelection<
                            mlpack::KMeans<mlpack::EuclideanDistance,
                                           mlpack::SampleInitialization,
                                           mlpack::KillEmptyClusters,
                                           mlpack::HamerlyKMeans>,
                            10 /* iterations */>> n(50);
n.Kernel().Bandwidth(100.0);

arma::mat g;
n.Apply(dataset, g);

// Print the first ten singular values of the approximation.
// Note that forming the approximate kernel matrix via `g * g.t()`
// would be very large, since the Corel dataset has many points!
arma::vec s = arma::svd(g);
std::cout << "First ten singular values of low-rank kernel matrix "
    << "approximation: " << s.subvec(0, 9).t();
```
