## `PCA`

The `PCA` class implements principal components analysis (PCA), a standard
machine learning data preparation technique.  PCA can be used to reduce the
number of dimensions in a dataset, or to preserve a certain percentage of the
variance of a dataset.

By default, `PCA` uses the full exact singular value decomposition (SVD), but
supports the use of other more efficient decompositions, including approximate
singular value decompositions.

#### Simple usage example:

```c++
// Use PCA to reduce the number of dimensions to 5 on uniform random data.

// This dataset is uniform random in 10 dimensions.
// Replace with a data::Load() call or similar for a real application.
arma::mat dataset(10, 1000, arma::fill::randu); // 1000 points.

mlpack::PCA pca;       // Step 1: create PCA object.
pca.Apply(dataset, 5); // Step 2: reduce data dimension to 5.

// Print some information about the modified dataset.
std::cout << "The transformed data matrix has size " << dataset.n_rows /* 5 */
    << " x " << dataset.n_cols << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructors](#constructors): create `PCA` objects.
 * [`Apply()`](#applying-transformations): apply PCA transformation to data.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.
 * [Template parameters](#advanced-functionality-different-decomposition-strategies)
   for using different decomposition strategies.

#### See also:

 * [`Radical`](radical.md): independent components analysis
 * [mlpack preprocessing utilities](../preprocessing.md)
 * [mlpack transformations](../transformations.md)
 * [Principal component analysis on Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)

### Constructors

 * `pca = PCA(scaleData=false)`
   - Construct a `PCA` object.
   - If `scaleData` is `true`, then all dimensions will have variance scaled to
     1 before applying PCA.
   - The `scaleData` parameter can be inspected with `pca.ScaleData()`, and also
     set; `pca.ScaleData() = true` will enable data variance scaling.

---

 * `pca = PCA(scaleData, decompositionPolicy)`
   - Construct a `PCA` object with a custom decomposition policy.
   - See the documentation for using
     [different decomposition strategies](#advanced-functionality-different-decomposition-strategies).

### Applying Transformations

 * `pca.Apply(data, transformedData)`
 * `pca.Apply(data, transformedData, eigVal)`
 * `pca.Apply(data, transformedData, eigVal, eigVec)`
   - Transform the
     [column-major matrix](../matrices.md#representing-data-in-mlpack) `data`
     using PCA, storing the result in `transformedData`.
   - `data` should be a floating-point matrix (e.g. `arma::mat`, `arma::fmat`,
     `arma::sp_mat`, etc.) or an expression that evaluates to one.
   - `transformedData` should be a dense floating-point matrix (e.g.,
     `arma::mat`, `arma::sp_mat`).
   - The size of `transformedData` will be the same as the size of `data`.
   - Dimensions in `transformedData` will be ordered decreasing in variance;
     that is, the first row of `transformedData` will correspond to the
     dimension with maximum variance.
   - Optionally, eigenvalues and eigenvectors of the covariance matrix can be
     returned:
     * If specified, `eigVal` should be a dense floating-point vector (e.g.
       `arma::vec`, `arma::fvec`, etc.) and will be filled with the eigenvalues
       of `transformedData`.
     * If specified, `eigvec` should be a dense floating-point matrix (e.g.
       `arma::mat`, `arma::fmat`, etc.) and will be filled with the eigenvectors
       of `transformedData`.

---

 * `double varRetained = pca.Apply(data, transformedData, newDimension)`
   - Use PCA to reduce the number of dimensions in the
     [column-major matrix](../matrices.md#representing-data-in-mlpack) `data`
     to `newDimension`, storing the result in `transformedData`.
   - `data` should be a floating-point matrix (e.g. `arma::mat`,
     `arma::fmat`, `arma::sp_mat`, etc.) or an expression that evaluates to
     one.
   - `transformedData` should be a dense floating-point matrix with the same
     element type as `data` (e.g. `arma::mat`, `arma::fmat`).
   - `transformedData` will have `newDimension` rows after the transformation.
   - Returns a `double` indicating the percentage of variance retained (between
     `0.0` and `1.0`).

---

 * `double varRetained = pca.Apply(data, transformedData, varianceToKeep)`
   - Use PCA to retain the dimensions of the
     [column-major matrix](../matrices.md#representing-data-in-mlpack) `data`
     that capture a factor of `varianceToKeep` of the data variance.
   - `data` should be a floating-point matrix (e.g. `arma::mat`, `arma::fmat`,
     `arma::sp_mat`, etc.) or an expression that evaluates to one.
   - `transformedData` should be a dense floating-point matrix with the same
     element type as `data` (e.g. `arma::mat`, `arma::fmat`).
   - `transformedData` will have `newDimension` rows after the transformation.
   - `varianceToKeep` should be a floating-point value between `0.0` and `1.0`.
     If `1.0`, all of the data variance is retained, and this is equivalent to
     the first version of `Apply()` (above).
   - Returns a `double` indicating the percentage of variance actually retained
     (between `0.0` and `1.0`).

---

 * `double varRetained = pca.Apply(data, newDimension)`
 * `double varRetained = pca.Apply(data, varianceToKeep)`
   - In-place versions of the two `Apply()` functions above.
   - Equivalent to `pca.Apply(data, data, newDimension)` or
     `pca.Apply(data, data, varianceToKeep)`.
   - `data` should be a dense floating-point matrix (e.g. `arma::mat`,
     `arma::fmat`, etc.).

---

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `PCA` class.

---

Apply PCA to a dataset, keeping dimensions that capture 90% of the data
variance.

```c++
// See https://datasets.mlpack.org/satellite.train.csv.
arma::mat data;
mlpack::data::Load("satellite.train.csv", data, true);
const size_t origDim = data.n_rows;

mlpack::PCA pca;

// Keep 90% of the data variance.
pca.Apply(data, 0.9);

std::cout << "PCA kept " << data.n_rows << " of " << origDim << " dimensions "
    << "to capture 90\% of the data variance." << std::endl;
```

---

Apply PCA to a 32-bit floating point dataset with dimension scaling, keeping all
dimensions, and printing the 5 largest eigenvalues of the covariance matrix of
the transformed data.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::fmat data;
mlpack::data::Load("iris.csv", data, true);

mlpack::PCA pca(true /* scale data when transforming */);

arma::fvec eigval;
arma::fmat transformedData;

pca.Apply(data, transformedData, eigval);

std::cout << "First point, before PCA: " << data.col(0).t();
std::cout << "First point, after PCA:  " << transformedData.col(0).t();
std::cout << std::endl;

// Now print the top 5 eigenvalues.
for (size_t i = 0; i < 5; ++i)
  std::cout << "Eigenvalue " << i << ": " << eigval[i] << "." << std::endl;
```

---

Apply PCA to a random sparse dataset, to reduce the dimensionality to a
20-dimensional dense dataset.

```c++
arma::sp_mat data;
// This dataset has 10k points in 1k dimensions, with 1% density.
data.sprandn(1000, 10000, 0.01);

mlpack::PCA pca(true /* scale data when transforming */);

arma::mat transformedData;
const double varianceRetained = pca.Apply(data, transformedData, 20);

std::cout << "First point, before PCA: " << data.col(0).t();
std::cout << "First point, after PCA: " << transformedData.col(0).t();

// Note that for random uniform data, this won't capture very much of the
// variance!  It would be much more for a real, structured dataset.
std::cout << "50 dimensions captured " << (100.0 * varianceRetained) << "\% of "
    << "the data variance." << std::endl;
```

---

### Advanced Functionality: Different Decomposition Strategies

By default, `PCA` uses the full exact singular value decomposition (SVD) to
transform data.  However, for very large datasets, it may be faster to use
alternative strategies, some of which may be approximate.  The `PCA` class has
one template parameter that allows different decomposition strategies to be
used.  The full signature of the class is:

```
PCA<DecompositionPolicy>
```

`DecompositionPolicy` specifies the strategy to be used to compute the singular
values and vectors of a data matrix.

Several decomposition policies are already implemented and ready for drop-in
usage:

 * `ExactSVDPolicy` _(default)_: use Armadillo's `svd()` and `svd_econ()`
   functions to compute the SVD
 * `RandomizedSVDPCAPolicy`: use the randomized SVD algorithm to compute the SVD
   <!-- TODO: add link to documentation! -->
 * `RandomizedBlockKrylovSVDPolicy`: use the randomized Block Krylov SVD
   algorithm to compute the SVD <!-- TODO: add link to documentation! -->
 * `QUICSVDPolicy`: use the tree-based `QUIC-SVD` algorithm to compute the SVD
   <!-- TODO: add link to documentation -->

The simple example program below uses all four decomposition types on the same
MNIST data, timing how long each decomposition takes.

```c++
arma::mat data;
// See https://datasets.mlpack.org/mnist.train.csv.
mlpack::data::Load("mnist.train.csv", data, true);

arma::mat output1, output2, output3, output4;

mlpack::PCA<mlpack::ExactSVDPolicy> pca1;
mlpack::PCA<mlpack::RandomizedSVDPCAPolicy> pca2;
mlpack::PCA<mlpack::RandomizedBlockKrylovSVDPolicy> pca3;
mlpack::PCA<mlpack::QUICSVDPolicy> pca4;

// Compute decompositions on all four, timing each one.
arma::wall_clock c;

c.tic();
pca1.Apply(data, output1);
const double pca1Time = c.toc();

c.tic();
pca2.Apply(data, output2);
const double pca2Time = c.toc();

c.tic();
pca3.Apply(data, output3);
const double pca3Time = c.toc();

c.tic();
pca4.Apply(data, output4);
const double pca4Time = c.toc();

std::cout << "PCA computation times for " << data.n_rows << " x " << data.n_cols
    << " data:" << std::endl;
std::cout << " - ExactSVDPolicy:                 " << pca1Time << "s."
    << std::endl;
std::cout << " - RandomizedSVDPCAPolicy:         " << pca2Time << "s."
    << std::endl;
std::cout << " - RandomizedBlockKrylovSVDPolicy: " << pca3Time << "s."
    << std::endl;
std::cout << " - QUICSVDPolicy:                  " << pca4Time << "s."
    << std::endl;
```

---

#### Custom decomposition policies

Instead of using the predefined classes above, it is also possible to implement
fully custom functionality via a new decomposition policy.  Any new
decomposition policy must implement one method:

```c++
class CustomDecompositionPolicy
{
 public:
  // Given input data `data` and `centeredData`, compute the singular value
  // decomposition of the data, and then project the data onto the first `rank`
  // singular vectors.
  //
  //  * `data` is the input matrix.  It is not guaranteed to be centered or
  //      scaled.
  //  * `centeredData` is the centered (and possibly scaled) version of the
  //      input matrix (e.g. the mean of each dimension is 0).
  //  * `transformedData` should be overwritten with the centered data's
  //      projection onto the singular vectors.
  //  * `svals` and `svecs` should be filled with the singular values and
  //      vectors of the centered data.
  //  * `rank` specifies the number of singular values/vectors to keep, and the
  //      dimension of `transformedData` should be equivalent to `rank`.  `rank`
  //      will be at most equal to `data.n_rows`.
  //
  //  * `InMatType` is a dense floating-point matrix type, but may be a subview
  //      or expression.
  //  * `MatType` is the type of matrix used to represent data, and will be a
  //      dense floating-point matrix type (e.g. `arma::mat`, `arma::fmat`,
  //      etc.).
  //  * `VecType` is the corresponding vector type to `MatType` (e.g., a
  //      `MatType` of `arma::mat` would mean a `VecType` of `arma::vec`, etc.).
  template<typename InMatType, typename MatType, typename VecType>
  static void Apply(const InMatType& data,
                    const MatType& centeredData,
                    MatType& transformedData,
                    VecType& svals,
                    MatType& svecs,
                    const size_t rank);
};
```
