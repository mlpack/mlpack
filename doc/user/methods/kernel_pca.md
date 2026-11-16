## `KernelPCA`: kernel principal components analysis

The `KernelPCA` class implements kernel principal components analysis, a
nonlinear data preprocessing technique.  Kernel PCA can be used for
nonlinear dimensionality reduction, or to project non-linearly-separable data
into a new space that may be linearly separable.  For example, kernel PCA with
the Gaussian kernel can project the two concentric circles below into a new
space where they are trivially separated by a classifier such as
[`LogisticRegression`](logistic_regression.md).

<!-- TODO: image with concentric circles before and after -->

mlpack's `KernelPCA` class allows the use of arbitrary kernels and, because
kernel PCA scales quadratically with data size, accelerated approximate variants
such as the Nystroem method are implemented and supported.

#### Simple usage example:

```c++
// Use KernelPCA to map the two concentric circles dataset into a linearly
// separable space.
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);

// Step 1: create object.
mlpack::KernelPCA<mlpack::GaussianKernel> kpca;

// Step 2: apply transform.
arma::mat transformedData;
kpca.Apply(data, transformedData, 2 /* new dimensionality */);

// Now any simple classifier or downstream technique could be applied.
std::cout << "Dimensionality of new data: " << transformedData.n_rows << "."
    << std::endl;
std::cout << "Bounding box of new data: [["
    << arma::min(transformedData.row(0)) << ", "
    << arma::min(transformedData.row(1)) << "], ["
    << arma::max(transformedData.row(0)) << ", "
    << arma::max(transformedData.row(1)) << "]]." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `KernelPCA` objects.
 * [`Apply()`](#applying-transformations): apply kernel PCA transformation to
   data.
 * [Other functionality](#other-functionality) for loading, saving, and
   inspecting.
 * [Examples](#simple-examples) of simple usage.
 * [Template parameters](#advanced-functionality-different-kernel-rules) for
   using different approximation strategies.
 * [Advanced examples](#advances-examples) that use the Nystroem method to
   efficiently approximate the kernel matrix on larger datasets.

#### See also:

 * [`PCA`](pca.md): principal components analysis
 * [mlpack kernels](../core/kernels.md): list of kernels implemented by mlpack
 * [`NystroemMethod`](nystroem_method.md): sampling-based approximation strategy
   for kernel matrices
 * [`RADICAL`](radical.md): independent components analysis
 * [mlpack preprocessing utilities](../preprocessing.md)
 * [mlpack transformations](../transformations.md)
 * [Kernel principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis)

### Constructors

 * `k = KernelPCA<KernelType>()`
   - Construct a `KernelPCA` object using the given `KernelType`.
   - `KernelType` must be either an [mlpack-supplied kernel](../core/kernels.md)
     or a [custom kernel class](../core/kernels.md#implement-a-custom-kernel).

 * `k = KernelPCA<KernelType>(kernel=KernelType(), centerTransformedData=false)`
   - Construct a `KernelPCA` object using the given `KernelType`, optionally
     passing an instantiated `KernelType` object and whether or not to center
     the transformed data around the origin.
   - `KernelType` must be either an [mlpack-supplied kernel](../core/kernels.md)
     or a [custom kernel class](../core/kernels.md#implement-a-custom-kernel).

 * `k = KernelPCA<KernelType, KernelRule>(kernel, centerTransformedData, rule=KernelRule())`
   - Construct a `KernelPCA` object using the given `KernelType` and
     [`KernelRule`](#advanced-functionality-template-parameters) strategy for
     kernel matrix computation, optionally
     passing instantiated `KernelType` and `KernelRule` objects, and whether or
     not to center the transformed data around the origin.
   - `KernelType` must be either an [mlpack-supplied kernel](../core/kernels.md)
     or a [custom kernel class](../core/kernels.md#implement-a-custom-kernel).
   - `KernelRule` must be one mlpack's pre-supplied kernel rules, or a custom
     kernel rule that matches the API described in the
     [advanced functionality section](#advanced-functionality-template-parameters).

#### Constructor parameters

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `KernelType` | [kernel class](../core/kernels.md) | Template parameter specifying which kernel to use for kernel PCA. | _(N/A)_ |
|----------|----------|-----------------|-------------|
| `kernel` | [`KernelType`](../core/kernels.md) | Instantiated kernel object to use when applying kernel PCA. | `KernelType()` |
| `centerTransformedData` | `bool` | If `true`, then transformed data will be centered around the origin such that `mean(X)` is the origin. | `false` |
| `rule` | [`KernelRule`](#advanced-functionality-template-parameters) | Instantiated strategy to use to compute or approximate the kernel matrix. | `KernelRule()` |

***Notes:***

 * Selecting the correct `KernelType` depends strongly on the data.  A decent
   default is the [`GaussianKernel`](../core/kernels.md#gaussiankernel),
   although its bandwidth needs to be tuned to the data.  But---depending on the
   distribution and shape of the data, different kernels may yield extremely
   different results---so [`GaussianKernel`](../core/kernels.md) may not always
   be the best.

 * If [`LinearKernel`](../core/kernels.md#linearkernel) is used, then kernel PCA
   is equivalent to regular PCA and the [`PCA`](pca.md) class can be used
   instead for more efficient results.

### Applying transformations

 * `k.Apply(data, transformedData, eigval)`
   - Apply kernel PCA to the given `data`, storing the result in
     `transformedData`, which will have size `data.n_cols` x `data.n_cols`.
   - Eigenvalues of the kernel matrix are stored in descending order in the
     vector `eigval`, which will be set to length `data.n_cols`.
   - If `data.n_cols` is large, consider one of the `Apply()` overloads below
     that takes a `newDimension` parameter, which results in a much smaller
     transformed data matrix.

 * `k.Apply(data, transformedData, eigval, eigvec)`
   - Apply kernel PCA to the given `data`, storing the result in
     `transformedData`, which will have size `data.n_cols` x `data.n_cols`.
   - Eigenvalues of the kernel matrix are stored in descending order in the
     vector `eigval`, which will be set to length `data.n_cols`.
   - Kernel matrix eigenvectors associated with each kernel eigenvalue are
     stored in the matrix `eigvec`, which will be set to size `data.n_cols` x
     `data.n_cols`.
   - The `i`th kernel eigenvector can be obtained with `eigvec.col(i)` and its
     corresponding kernel eigenvalue is `eigval[i]`.
   - If `data.n_cols` is large, consider one of the `Apply()` overloads below
     that takes a `newDimension` parameter, which results in a much smaller
     transformed data matrix.

 * `k.Apply(data, transformedData, eigval, eigvec, newDimension)`
   - Apply kernel PCA to the given `data`, keeping only the top `newDimension`
     kernel eigendimensions.
   - The resulting transformed data is stored in `transformedData`, and will
     have size `newDimension` x `data.n_cols`.
   - The top `newDimension` eigenvalues of the kernel matrix are stored
     descending order in the `eigval` vector, which will be set to length
     `newDimension`.
   - The top `newDimension` kernel eigenvectors associated with each kernel
     eigenvalue are stored in the matrix `eigvec`, which will be set to size
     `newDimension` x `data.n_cols`.
   - The `i`th kernel eigenvector can be obtained with `eigvec.col(i)` and its
     corresponding kernel eigenvalue is `eigval[i]`.

 * `k,Apply(data, newDimension)`
   - Apply kernel PCA in-place to `data`, keeping only the top `newDimension`
     kernel eigendimensions.
   - When finished, `data` will be set to size `newDimension` x `data.n_cols`.

#### `Apply()` Parameters:

| **name** | **type** | **description** |
|-----------|----------|----------|-----------------|
| `data` | [`arma::mat`](../matrices.md) | Input dataset to perform kernel PCA on. |
| `transformedData` | [`arma::mat`](../matrices.md) | Matrix to store transformed data into; will be filled with data of size `data.n_cols` x `data.n_cols`, or `newDimension` x `data.n_cols` if `newDimension` was specified. |
| `eigval` | [`arma::vec&`](../matrices.md) | Vector to store kernel eigenvalues
into (in descending order). |
| `eigvec` | [`arma::mat&`](../matrices.md) | Matrix to store kernel eigenvectors into. |
| `newDimension` | `size_t` | Desired new dimensionality of data (e.g. number of kernel eigenvectors to keep). |

***Notes:***

 * Kernel PCA scales *quadratically* in the number of points (`data.n_cols`).
   If your dataset is large, `Apply()` may exhaust the system's memory or take a
   very long time.  In that case, the use of an approximate kernel matrix rule
   as the `KernelRule` template parameter may be helpful; see the
   [advanced functionality section](#advanced-functionality-template-parameters).

 * Different element types can be used for `data` (e.g. `arma::fmat` or any
   dense matrix type implementing the Armadillo API).  If a different type is
   used for `data`, then the types of `transformedData`, `eigval`, and `eigvec`
   will match the given element type.  For example, if `data` is `arma::fmat`,
   then `transformedData` and `eigvec` must have type `arma::fmat`, and `eigval`
   must have type `arma::fvec`.

### Other functionality

 * `k.Kernel()` returns the `KernelType` that will be used for computing the
   kernel matrix.  If a `KernelType` was passed to the
   [constructor](#constructors), this is the same object.

 * `k.CenterTransformedData()` returns a `bool` indicating whether or not the
   transformed data will be centered about the origin;
   `k.CenterTransformedData() = c` can be used to change the setting.

 * A `KernelPCA` object can be serialized with
   [`Save()` and `Load()`](../load_save.md#mlpack-models-and-objects).

### Examples

Use kernel PCA on the concentric circles dataset with a downstream logistic
regression classifier; print the classification results before and after kernel
PCA.

```c++
// Load the datasets.
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);
arma::Row<size_t> labels;
mlpack::Load("http://datasets.mlpack.org/circles.labels.csv", labels,
    mlpack::Fatal);

// Perform kernel PCA on the dataset with default parameters.
mlpack::KernelPCA<mlpack::GaussianKernel> kpca;
arma::mat transformedData;
kpca.Apply(data, transformedData, 2);

// Now construct two classifiers: one on the original dataset, and one on the
// transformed dataset.
mlpack::LogisticRegression lr1, lr2;
lr1.Train(data, labels);
lr2.Train(transformedData, labels);

// Compute the correctness of each on the training data.  The preprocessed
// dataset will have perfect results because it has been mapped to a linearly
// separable space!
arma::Row<size_t> preds1, preds2;
lr1.Classify(data, preds1);
lr2.Classify(transformedData, preds2);

std::cout << "Logistic regression correct on original dataset:    "
    << arma::accu(preds1 == labels) << " of " << labels.n_elem << "."
    << std::endl;
std::cout << "Logistic regression correct on transformed dataset: "
    << arma::accu(preds2 == labels) << " of " << labels.n_elem << "."
    << std::endl;
```

---

Perform kernel PCA with the Gaussian kernel with a specified bandwidth on the
concentric circles dataset, and print some eigenvalues of the kernel matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/circles.csv", data, mlpack::Fatal);

// Create the kernel with a bandwidth of 0.75, and then the kernel PCA object.
mlpack::GaussianKernel gk(0.75);
mlpack::KernelPCA<mlpack::GaussianKernel> kpca(gk);

// Transform the data, returning the eigenvalues and eigenvectors.
arma::mat transformedData, eigenvectors;
arma::vec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors);

// Print some eigenvalues and eigenvectors.
std::cout << "Total number of eigenvalues: " << eigenvalues.n_elem << "."
    << std::endl;
std::cout << "Size of eigenvectors matrix: " << eigenvectors.n_rows << " x "
    << eigenvectors.n_cols << "." << std::endl;
std::cout << "First five eigenvalues: " << eigenvalues.subvec(0, 4).t();
```

---

Perform kernel PCA on the cloud dataset using the cosine similarity as the
kernel function, mapping down to three dimensions.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", data, mlpack::Fatal);

// Use the cosine similarity as the kernel function.
mlpack::KernelPCA<mlpack::CosineSimilarity> kpca;

arma::mat transformedData;
kpca.Apply(data, transformedData, 3 /* dimensions */);

// Print the first three points before and after.
for (size_t i = 0; i < 3; ++i)
{
  std::cout << "Point " << i << ":" << std::endl;
  std::cout << " - Before KPCA: " << data.col(i).t();
  std::cout << " - After KPCA:  " << transformedData.col(i).t();
}
```

---

Perform kernel PCA on the satellite dataset using the Laplacian kernel with a
custom bandwidth, and using 32-bit floating point values to represent the data.

```c++
arma::fmat data;
mlpack::Load("http://datasets.mlpack.org/satellite.train.csv", data,
    mlpack::Fatal);

// Create a KernelPCA object with the LaplacianKernel, then modify its
// bandwidth.
mlpack::KernelPCA<mlpack::LaplacianKernel> kpca;
kpca.Kernel().Bandwidth(50.0);

// Compute kernel PCA, keeping the top 3 dimensions and their eigenvalues and
// eigenvectors.
arma::fmat transformedData, eigenvectors;
arma::fvec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors, 3);

std::cout << "Transformed data size: " << transformedData.n_rows << " x "
    << transformedData.n_cols << "." << std::endl;
std::cout << "First three eigenvalues: " << eigenvalues.t();
```

---

### Advanced functionality: different kernel rules

The `KernelPCA` class has two template parameters that can be used for custom
behavior.  The full signature of the class is:

```
KernelPCA<KernelType, KernelRule>
```

 * `KernelType` is the type of kernel function to use when computing the kernel
   matrix.  There are many [mlpack kernels](../core/kernels.md) available for
   drop-in usage, and
   [implementing a custom kernel](../core/kernels.md#implement-a-custom-kernel)
   is quite straightforward.  Common choices of kernels include:
   - [`GaussianKernel`](../core/kernels.md#gaussiankernel)
   - [`CauchyKernel`](../core/kernels.md#cauchykernel)
   - [`LaplacianKernel`](../core/kernels.md#laplaciankernel)
   - [`PolynomialKernel`](../core/kernels.md#polynomialkernel)
   - [`SphericalKernel`](../core/kernels.md#sphericalkernel)

 * `KernelRule` specifies the strategy to be used to compute the kernel matrix.
   Because kernel PCA depends on the computation (or approximation) of the
   kernel matrix, and the kernel matrix has size `data.n_cols` x `data.n_cols`,
   kernel PCA scales very poorly to large datasets.  The use of a custom
   `KernelRule` can help mitigate this problem.
   - `NaiveKernelRule` (the default) simply computes the kernel matrix as a
     `data.n_cols` x `data.n_cols` matrix, by computing the kernel function
     between each pair of points.

   - `NystroemKernelRule<>` is a customizable approach based on
     [`NystroemMethod`](nystroem_method.md) that scales significantly better, by
     taking a small sample of points from the dataset and using a low-rank
     approximation of the sampled kernel matrix.
     * By default, 5 iterations of [`KMeans`](kmeans.md) is used to compute a
       sample of `0.1 * data.n_cols` centroids.
     * The `NystroemKernelRule<>` class is configurable; see
       [its documentation below](#nystroemkernelrule).

   - A custom kernel rule is straightforward to implement and must implement the
     following API:

```c++
class CustomKernelRule
{
  // A default constructor must be available.
  CustomKernelRule();

  // Compute or approximate the kernel matrix of `data`, using the given kernel
  // `kernel`.  The approximate eigenvalues and eigenvectors should be computed
  // and stored in `eigval` and `eigvec`, respectively, and the data should be
  // projected onto those eigenvectors and stored into `transformedData`.
  //
  // Note that a kernel rule does not need to compute *every* eigenvector and
  // eigenvalue of the kernel matrix, and may choose to only compute some.  That
  // is up to you as an implementer.
  //
  // Supposing that the class computes `r` eigenvalues and eigenvectors:
  //
  //  - the length of `eigval` is expected to be `r`;
  //  - the size of `eigvec` is expected to be `r` x `r`;
  //  - the size of `transformedData` is expected to be `data.n_cols` x `r`.
  //
  // `MatType` will be a dense matrix type that matches the Armadillo API (e.g.
  // `arma::mat`, and `VecType` will be the corresponding column vector type
  // (e.g.  `arma::vec`).
  template<typename MatType, typename VecType, typename KernelType>
  void ApplyKernelMatrix(const MatType& data,
                         MatType& transformedData,
                         VecType& eigval,
                         MatType& eigvec,
                         KernelType kernel);

  // The class must be serializable.
  //
  // If you never plan to serialize a KernelPCA that uses your kernel rule, you
  // can leave the function body empty.  Otherwise, see the cereal
  // documentation, or follow the example of other mlpack methods.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int version);
};
```

---

#### `NystroemKernelRule`

The `NystroemKernelRule<>` class uses the [`NystroemMethod`](nystroem_method.md)
class to compute a low-rank approximation to the kernel matrix using a small
number of samples from the dataset.  Different sample selection strategies can
be configured via template parameter, and the size of the sample can be
configured as a constructor parameter.

 * `NystroemKernelRule<KMeansSelection<>>` or `NystroemKernelRule<>` can be used
   to compute samples as centroids of a few iterations of [`KMeans`](kmeans.md)
   clustering.
    - The `KMeansSelection<>` class has two template parameters:
      `KMeansSelection<KMeansType, numIterations>`.
    - `KMeansType` can be any templated version of the [`KMeans`](kmeans.md)
      class.
    - `numIterations` is a `size_t` representing how many iterations of k-means
      to run.

 * `NystroemKernelRule<RandomSelection>` will select points randomly from the
   dataset.

 * `NystroemKernelRule<OrderedSelection>` will select the first points from the
   dataset.

 * A custom selection strategy can be implemented; the requirements are
   described in the [`NystroemMethod`](nystroem_method.md) documentation.

---

A `NystroemKernelRule<>` object has the following two constructors:

 * `n = NystroemKernelRule(datasetPct=0.01)`
 * `n = NystroemKernelRule<SelectionStrategy>(datasetPct=0.01)`
   - Construct a `NystroemKernelRule`, optionally with a specific
     `SelectionStrategy`.
   - The given percentage of the dataset size will be used as the sample size.

 * `n = NystroemKernelRule(numPoints)`
 * `n = NystroemKernelRule<SelectionStrategy>(numPoints)`
   - Construct a `NystroemKernelRule`, optionally with a specific
     `SelectionStrategy`.
   - The given number of points will be used as the sample size.

### Advanced examples

These examples use custom `KernelRule`s to approximate the kernel matrix and
scale kernel PCA to larger datasets.

---

Perform kernel PCA on the cloud dataset, using the
[`NystroemKernelRule`](#nystroemkernelrule) to approximate the kernel matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/cloud.csv", data, mlpack::Fatal);

// Use the Nystroem method with default parameters to approximate the kernel
// matrix.
mlpack::KernelPCA<mlpack::GaussianKernel, mlpack::NystroemKernelRule<> > kpca;
kpca.Kernel().Bandwidth(250.0);

arma::mat transformedData;
kpca.Apply(data, transformedData, 3 /* dimensions */);

// Print the first five data points.
std::cout << "First five data points:" << std::endl;
for (size_t i = 0; i < 5; ++i)
{
  std::cout << " - Point " << i << ": " << transformedData.col(i).t();
}
```

---

Perform kernel PCA on the corel dataset, using 100 randomly selected points with
the [`NystroemKernelRule`](#nystroemkernelrule) to approximate the kernel
matrix.

```c++
arma::mat data;
mlpack::Load("http://datasets.mlpack.org/corel-histogram.csv", data,
    mlpack::Fatal);

// Instantiate the Nystroem kernel rule so that we can set the number of points
// to use to 100 instead of the default of 1% of the dataset.
mlpack::NystroemKernelRule<mlpack::RandomSelection> nkr(100);
mlpack::KernelPCA<mlpack::GaussianKernel,
                  mlpack::NystroemKernelRule<mlpack::RandomSelection>> kpca(
    mlpack::GaussianKernel(), true, nkr);

// Transform the data, keep the top 5 dimensions, and get the eigenvalues.
arma::mat transformedData, eigenvectors;
arma::vec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors, 5);

// Print the eigenvalues.
std::cout << "First 5 eigenvalues of approximated kernel matrix: "
    << eigenvalues.t();
```

---

Perform kernel PCA on the corel dataset using 32-bit floats to represent the
data.  Use [`ElkanKMeans`](kmeans.md#lloydsteptype) to efficiently perform 10
iterations of k-means, producing 50 centroids to approximate the data.

```c++
arma::fmat data;
mlpack::Load("http://datasets.mlpack.org/corel-histogram.csv", data,
    mlpack::Fatal);

// Instantiate the Nystroem kernel rule to use k-means with the Elkan strategy,
// computing a number of centroids equal to 2% of the data points.
using KernelRuleType = mlpack::NystroemKernelRule<mlpack::KMeansSelection<
    mlpack::KMeans<mlpack::EuclideanDistance,
                   mlpack::SampleInitialization,
                   mlpack::MaxVarianceNewCluster,
                   mlpack::ElkanKMeans>, 10 /* iterations */>>;
KernelRuleType nkr(0.02);
mlpack::KernelPCA<mlpack::GaussianKernel, KernelRuleType> kpca(
    mlpack::GaussianKernel(), true, nkr);

// Transform the data, keep the top 5 dimensions, and get the eigenvalues.
arma::fmat transformedData, eigenvectors;
arma::fvec eigenvalues;
kpca.Apply(data, transformedData, eigenvalues, eigenvectors, 5);

// Print the eigenvalues.
std::cout << "First 5 eigenvalues of approximated kernel matrix: "
    << eigenvalues.t();
```
