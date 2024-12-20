## `Radical`

The `Radical` class implements RADICAL, the ***R***obust, ***A***ccurate,
***D***irect ***I***ndependent ***C***omponents ***A***nalysis (ICA)
a***L***gorithm.  ICA can be used to transform a matrix `X` into a new matrix
`Y` where each of the rows of `Y` are independent components.  ICA also recovers
a square "mixing matrix" `W`, such that `Y = W * X`.  mlpack's implementation of
RADICAL supports decomposing different matrix types via template parameters.

#### Simple usage example:

```c++
// Use RADICAL to convert the matrix into one where each dimension is
// linearly independent.

// This dataset is uniform random in 3 dimensions.
// Replace with a data::Load() call or similar for a real application.
arma::mat x(3, 100, arma::fill::randu); // 1000 points.

mlpack::Radical r; // Step 1: create RADICAL object.
arma::mat w, y;
r.Apply(x, y, w);  // Step 2: perform RADICAL on data.

// Print some information about the mixing matrix.
std::cout << "Mixing matrix size: " << w.n_rows << " x " << w.n_cols << "."
    << std::endl;

// Print some information about the transformed matrix.
std::cout << "Independent components matrix size: " << y.n_rows << " x "
    << y.n_cols << "." << std::endl;
```
<p style="text-align: center; font-size: 85%"><a href="#simple-examples">More examples...</a></p>

#### Quick links:

 * [Constructor](#constructor): create `Radical` objects.
 * [`Apply()`](#applying-transformations): apply RADICAL transformation to data.
 * [Serialization](#serialization) for loading and saving `Radical` objects.
 * [Examples](#simple-examples) of simple usage and links to detailed example
   projects.

#### See also:

 * [`PCA`](pca.md): principal components analysis
 * [mlpack preprocessing utilities](../preprocessing.md)
 * [mlpack transformations](../transformations.md)
 * [ICA Using Spacings Estimates of Entropy (pdf)](https://www.jmlr.org/papers/volume4/learned-miller03a/learned-miller03a.pdf)
 * [Independent components analysis on Wikipedia](https://en.wikipedia.org/wiki/Independent_component_analysis)

### Constructor

 * `r = Radical()`
 * `r = Radical(noiseStdDev=0.175, replicates=30, angles=150, sweeps=0, m=0)`
   - Construct a `Radical` object with the given parameters.

---

#### Constructor Parameters:

| **name** | **type** | **description** | **default** |
|----------|----------|-----------------|-------------|
| `noiseStdDev` | `double` | Standard deviation of Gaussian noise to add to the data. | `0.175` |
| `replicates` | `size_t` | Number of Gaussian-perturbed replicates to use (per point). | `30` |
| `angles` | `size_t` | Number of angles to consider in brute-force search during 2-D RADICAL. | `150` |
| `sweeps` | `size_t` | Number of sweeps.  Each sweep calls 2-D RADICAL once for each pair of dimensions.  `0` will set sweeps to the number of dimensions in the data minus one. | `0` |
| `m` | `size_t` | The variable `m` from Vasicek's m-spacing estimator of entropy (see [Eq. (3)](https://www.jmlr.org/papers/volume4/learned-miller03a/learned-miller03a.pdf)).  `0` will use the square root of the number of dimensions in the data. | `0` |

As an alternative to passing `noiseStdDev`, `replicates`, `angles`, `sweeps`,
and `m`, they can each be set or accessed with standalone methods:

 * `r.NoiseStdDev() = n` will set the standard deviation of the Gaussian noise
   to add to data to `n`.
 * `r.Replicates() = reps` will set the number of Gaussian-perturbed replicates
   to use per point to `reps`.
 * `r.Angles() = a` will set the number of angles to consider in brute-force
   search to `a`.
 * `r.Sweeps() = s` will set the number of sweeps to `s`.
 * `r.M() = m` will set the value of m to use for Vasicek's m-spacing estimator
   of entropy to `m`.

---

### Applying Transformations

 * `r.Apply(x, y, w)`
   - Apply RADICAL to the
     [column-major matrix](../matrices.md#representing-data-in-mlpack) `x`,
     storing the learned whitening matrix in `w` and learned independent
     components in `y`.
   - `w` will be set to size `x.n_rows` by `x.n_rows`.
   - `y` will be set to the same size as `x`.
   - `x` can be recovered as `w * y`.
   - `x`, `y`, and `w` should be dense floating-point matrix types (e.g.
     `arma::mat`, `arma::fmat`).  Any dense floating-point matrix type
     implementing the Armadillo API can be used.

***Note***: `Radical.Apply()` scales quadratically in the number of dimensions
of the data; so, when `x.n_rows` is high, `Radical.Apply()` may take a long
time!

---

### Serialization

 * A `Radical` object can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).
   Only the parameters to be used when calling `Apply()` are serialized (e.g.
   the five constructor parameters.)

---

### Simple Examples

See also the [simple usage example](#simple-usage-example) for a trivial usage
of the `Radical` class.

---

Apply RADICAL to the `iris` dataset.  Print the reconstruction error and
magnitude of each dimension of the RADICAL-ized matrix.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::mat dataset;
mlpack::data::Load("iris.csv", dataset);

// Create RADICAL object with default options and apply to data.
mlpack::Radical r;
arma::mat unmixingMatrix, independentDataset;
r.Apply(dataset, independentDataset, unmixingMatrix);

// Print the size of the new independent components dataset.
std::cout << "Size of transformed data: " << independentDataset.n_rows << " x "
    << independentDataset.n_cols << "." << std::endl;

// Print the reconstruction error.
const double reconError =
    arma::norm(independentDataset - unmixingMatrix * dataset, "F");
std::cout << "Reconstruction error: " << reconError << "." << std::endl;

// Print the magnitude of each dimension before and after RADICAL.
std::cout << "Dimension magnitudes before RADICAL:" << std::endl;
for (size_t i = 0; i < dataset.n_rows; ++i)
{
  std::cout << " - Dimension " << i << ": " << arma::norm(dataset.row(i)) << "."
      << std::endl;
}

std::cout << std::endl;
std::cout << "Dimension magnitudes after RADICAL:" << std::endl;
for (size_t i = 0; i < independentDataset.n_rows; ++i)
{
  std::cout << " - Dimension " << i << ": "
      << arma::norm(independentDataset.row(i)) << "." << std::endl;
}
```

---

Apply RADICAL to the `iris` dataset using a 32-bit floating point
representation, and confirm that the independent components are actually
independent.

```c++
// See https://datasets.mlpack.org/iris.csv.
arma::fmat dataset;
mlpack::data::Load("iris.csv", dataset);

// Create RADICAL object with custom options and apply to data.
mlpack::Radical r(0.1 /* noise standard deviation */,
                  25 /* replicates */,
                  120 /* angles */,
                  15 /* sweeps */,
                  5 /* m */);
arma::fmat unmixingMatrix, independentDataset;
r.Apply(dataset, independentDataset, unmixingMatrix);

// Check the linear independence of the resulting dimensions.
arma::fmat covOrig = arma::cov(dataset.t());
arma::fmat covRadical = arma::cov(independentDataset.t());

std::cout << "Covariance matrix of original data:" << std::endl;
std::cout << covOrig << std::endl;

std::cout << "Covariance matrix of data after RADICAL:" << std::endl;
std::cout << covRadical;
```

---

Apply RADICAL to a subset of the `iris` dataset, and then use the unmixing
matrix to apply the same transformation to a test set.

```c++
// See https://datasets.mlpack.org/iris.train.csv.
arma::mat trainSet;
mlpack::data::Load("iris.train.csv", trainSet, true);
// See https://datasets.mlpack.org/iris.test.csv.
arma::mat testSet;
mlpack::data::Load("iris.test.csv", testSet, true);

// Create RADICAL object with custom options.  Here we optimize for speed, but
// at the potential loss of quality!  A real-world application may want to use
// higher numbers of replicates and sweeps.
mlpack::Radical r;
r.NoiseStdDev() = 0.2;
r.Replicates() = 5; // Reduce number of replicates to keep things fast.
r.Sweeps() = 5; // Reduce number of sweeps to keep things fast.

arma::mat unmixing, trainIcs;

r.Apply(trainSet, trainIcs, unmixing);

// Now apply the unmixing matrix to the test set.
arma::mat testIcs = unmixing * testSet;

// Print some statistics about the training and test sets.  The average
// correlation between dimensions in the test sets may be higher than the
// training set (where the dimensions should be fully independent).
arma::mat covTrain = arma::cov(trainIcs.t());
arma::mat covTest = arma::cov(testIcs.t());

std::cout << "Covariance matrix of training data after RADICAL:" << std::endl;
std::cout << covTrain << std::endl;

std::cout << "Covariance matrix of test data after RADICAL:" << std::endl;
std::cout << covTest;

// After this point it would be possible to use any mlpack classifier on the
// unmixed datasets.
```
