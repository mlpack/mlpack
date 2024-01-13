# mlpack core class documentation

Underlying the implementations of [mlpack's machine learning
algorithms](index.md#mlpack-algorithm-documentation) are mlpack core support
classes, each of which are documented on this page.

 * [Core math utilities](#core-math-utilities): utility classes for mathematical
   purposes
 * [Distributions](#distributions): probability distributions
 * [Metrics](#metrics): distance metrics for geometric algorithms
 * [Kernels](#kernels): Mercer kernels for kernel-based algorithms

## Core math utilities

Utilities in the `mlpack::math::` namespace are meant to provide additional
mathematical support on top of Armadillo.

 * [`math::Range`](#mathrange): simple mathematical range (i.e. `[0, 3]`)

---

### `math::Range`

The `math::Range` class represents a simple mathematical range (i.e. `[0, 3]`),
with the bounds represented as `double`s.

---

#### Constructors

 * `r = math::Range()`
   - Construct an empty range.

 * `r = math::Range(p)`
   - Construct the range `[p, p]`.

 * `r = math::Range(lo, hi)`
   - Construct the range `[lo, hi]`.

---

#### Accessing and modifying range properties

 * `r.Lo()` and `r.Hi()` return the lower and upper bounds of the range as
   `double`s.
   - A range is considered empty if `r.Lo() > r.Hi()`.
   - These can be used to modify the bounds, e.g., `r.Lo() = 3.0`.

 * `r.Width()` returns the span of the range (i.e. `r.Hi() - r.Lo()`) as a
   `double`.

 * `r.Mid()` returns the midpoint of the range as a `double`.

---

#### Working with ranges

 * Given two ranges `r1` and `r2`,
   - `r1 | r2` returns the union of the ranges,
   - `r1 |= r2` expands `r1` to include the range `r2`,
   - `r1 & r2` returns the intersection of the ranges (possibly an empty range),
   - `r1 &= r2` shrinks `r1` to the intersection of `r1` and `r2`,
   - `r1 == r2` returns `true` if the two ranges are strictly equal (i.e. lower
     and upper bounds are equal),
   - `r1 != r2` returns `true` if the two ranges are not strictly equal,
   - `r1 < r2` returns `true` if `r1.Hi() < r2.Lo()`,
   - `r1 > r2` returns `true` if `r1.Lo() > r2.Hi()`, and
   - `r1.Contains(r2)` returns `true` if the ranges overlap at all.

 * Given a range `r` and a `double` scalar `d`,
   - `r * d` returns a new range `[d * r.Lo(), d * r.Hi()]`,
   - `r *= d` scales `r.Lo()` and `r.Hi()` by `d`, and
   - `r.Contains(d)` returns `true` if `d` is contained in the range.

---

 * To use ranges with different element types (e.g. `float`), use the type
   `math::RangeType<float>` or similar.

---

Example:

```c++
mlpack::math::Range r1(5.0, 6.0); // [5, 6]
mlpack::math::Range r2(7.0, 8.0); // [7, 8]

mlpack::math::Range r3 = r1 | r2; // [5, 8]
mlpack::math::Range r4 = r1 & r2; // empty range

bool b1 = r1.Contains(r2); // false
bool b2 = r1.Contains(5.5); // true
bool b3 = r1.Contains(r3); // true
bool b4 = r3.Contains(r4); // false

// Create a range of `float`s and a range of `int`s.
mlpack::math::RangeType<float> r5(1.0f, 1.5f); // [1.0, 1.5]
mlpack::math::RangeType<int> r6(3, 4); // [3, 4]
```

---

`math::Range` is used by:

 * [`RangeSearch`](range_search.md)
 * [mlpack trees](#trees)

---

## Distributions

mlpack has support for a number of different distributions, each supporting the
same API.  These can be used with, for instance, the [`HMM`](hmm.md) class.

 * [`DiscreteDistribution`](#discretedistribution): multidimensional categorical
   distribution (generalized Bernoulli distribution)
 * [`GaussianDistribution`](#gaussiandistribution): multidimensional Gaussian
   distribution

### `DiscreteDistribution`

`DiscreteDistribution` represents a multidimensional categorical distribution
(or generalized Bernoulli distribution) where integer-valued vectors (e.g.
`[0, 3, 4]`) are associated with specific probabilities in each dimension.

*Example:* a 3-dimensional `DiscreteDistribution` will have a specific
probability value associated with each integer value in each dimension.  So, for
the vector `[0, 3, 4]`, `P(0)` in dimension 0 could be, e.g., `0.3`, `P(3)` in
dimension 1 could be, e.g., `0.4`, and `P(4)` in dimension 2 could be, e.g.,
`0.6`.  Then, `P([0, 3, 4])` would be `0.3 * 0.4 * 0.6 = 0.072`.

---

#### Constructors

 * `d = DiscreteDistribution(numObservations)`
   - Create a one-dimensional discrete distribution with `numObservations`
     different observations in the one and only dimension.  `numObservations` is
     of type `size_t`.

 * `d = DiscreteDistribution(numObservationsVec)`
   - Create a multidimensional discrete distribution with
     `numObservationsVec.n_elem` dimensions and `numObservationsVec[i]`
     different observations in dimension `i`.
   - `numObservationsVec` is of type `arma::Col<size_t>`.

 * `d = DiscreteDistribution(probabilities)`
   - Create a multidimensional discrete distribution with the given
     probabilities.
   - `probabilities` should have type `std::vector<arma::vec>`, and
     `probabilities.size()` should be equal to the dimensionality of the
     distribution.
   - `probabilities[i]` is a vector such that `probabilities[i][j]` contains the
     probability of `j` in dimension `i`.

---

#### Access and modify properties of distribution

 * `d.Dimensionality()` returns a `size_t` indicating the number of dimensions
   in the multidimensional discrete distribution.

 * `d.Probabilities(i)` returns an `arma::vec&` containing the probabilities of
   each observation in dimension `i`.
   - `d.Probabilities(i)[j]` is the probability of `j` in dimension `i`.
   - This can be used to modify probabilities: `d.Probabilities(0)[1] = 0.7`
     sets the probability of observing the value `1` in dimension `0` to `0.7`.
   - *Note:* when setting probabilities manually, be sure that the sum of
     probabilities in a dimension is 1!

---

#### Compute probabilities of points

 * `d.Probability(observation)` returns the probability of the given
   observation as a `double`.
   - `observation` should be an `arma::vec` of size `d.Dimensionality()`.
   - `observation[i]` should take integer values between `0` and
     `d.Probabilities(i).n_elem - 1`.

 * `d.Probability(observations, probabilities)` computes the probabilities of
   many observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `d.Probability(observations.col(i))`.

 * `d.LogProbability(observation)` returns the log-probability of the given
   observation as a `double`.

 * `d.LogProbability(observations, probabilities)` computes the
   log-probabilities of many observations.

---

#### Sample from the distribution

 * `d.Random()` returns an `arma::vec` with a random sample from the
   multidimensional discrete distribution.

---

#### Fit the distribution to observations

 * `d.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `observations(j, i)` should be an integer value between `0` and the number
     of observations for dimension `i`.

 * `d.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `d`.

---

*Example usage:*

```c++
// Create a single-dimension Bernoulli distribution: P([0]) = 0.3, P([1]) = 0.7.
mlpack::DiscreteDistribution bernoulli(2);
bernoulli.Probabilities(0)[0] = 0.3;
bernoulli.Probabilities(0)[1] = 0.7;

const double p1 = bernoulli.Probability(arma::vec("0")); // p1 = 0.3.
const double p2 = bernoulli.Probability(arma::vec("1")); // p2 = 0.7.

// Create a 3-dimensional discrete distribution by specifying the probabilities
// manually.
arma::vec probDim0 = arma::vec("0.1 0.3 0.5 0.1"); // 4 possible values.
arma::vec probDim1 = arma::vec("0.7 0.3");         // 2 possible values.
arma::vec probDim2 = arma::vec("0.4 0.4 0.2");     // 3 possible values.
std::vector<arma::vec> probs { probDim0, probDim1, probDim2 };
mlpack::DiscreteDistribution d(probs);

arma::vec obs("2 0 1");
const double p3 = d.Probability(obs); // p3 = 0.5 * 0.7 * 0.4 = 0.14.

// Estimate a 10-dimensional discrete distribution.
// Each dimension takes values between 0 and 9.
arma::mat observations = arma::randi<arma::mat>(10, 1000,
    arma::distr_param(0, 9));

// Create a distribution with 10 observations in each of the 10 dimensions.
mlpack::DiscreteDistribution d2(
    arma::Col<size_t>("10 10 10 10 10 10 10 10 10 10"));
d2.Train(observations);

// Compute the probabilities of each point.
arma::vec probabilities;
d2.Probability(observations, probabilities);
std::cout << "Average probability: " << arma::mean(probabilities) << "."
    << std::endl;
```

---

### `GaussianDistribution`

`GaussianDistribution` is a standard multivariate Gaussian distribution with
parameterized mean and covariance.

---

#### Constructors

 * `g = GaussianDistribution(dimensionality)`
   - Create the distribution with the given dimensionality.
   - The distribution will have a zero mean and unit diagonal covariance matrix.

 * `g = GaussianDistribution(mean, covariance)`
   - Create the distribution with the given mean and covariance.
   - `mean` is of type `arma::vec` and should have length equal to the
     dimensionality of the distribution.
   - `covariance` is of type `arma::mat`, and should be symmetric and square,
     with rows and columns equal to the dimensionality of the distribution.

---

#### Access and modify properties of distribution

 * `g.Dimensionality()` returns the dimensionality of the distribution as a
   `size_t`.

 * `g.Mean()` returns an `arma::vec&` holding the mean of the distribution.
   This can be modified.

 * `g.Covariance()` returns a `const arma::mat&` holding the covariance of the
   distribution.  To set a new covariance, use `g.Covariance(newCov)` or
   `g.Covariance(std::move(newCov))`.

 * `g.InvCov()` returns a `const arma::mat&` holding the precomputed inverse of
   the covariance.

 * `g.LogDetCov()` returns a `double` holding the log-determinant of the
   covariance.

---

#### Compute probabilities of points

 * `g.Probability(observation)` returns the probability of the given
   observation as a `double`.
   - `observation` should be an `arma::vec` of size `d.Dimensionality()`.

 * `g.Probability(observations, probabilities)` computes the probabilities of
   many observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `g.Probability(observations.col(i))`.

 * `g.LogProbability(observation)` returns the log-probability of the given
   observation as a `double`.

 * `g.LogProbability(observations, probabilities)` computes the
   log-probabilities of many observations.

---

#### Sample from the distribution

 * `g.Random()` returns an `arma::vec` with a random sample from the
   multidimensional discrete distribution.

---

#### Fit the distribution to observations

 * `g.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `d.Dimensionality()`; `observations.n_cols` is the number of observations.

 * `g.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `d`.

---

*Example usage:*

```c++
// Create a Gaussian distribution in 3 dimensions with zero mean and unit
// covariance.
mlpack::GaussianDistribution g(3);

// Compute the probability of the point [0, 0.5, 0.25].
const double p = g.Probability(arma::vec("0 0.5 0.25"));

// Modify the mean in dimension 0.
g.Mean()[0] = 0.5;

// Set a random covariance.
arma::mat newCov(3, 3, arma::fill::randu);
newCov *= newCov.t(); // Ensure covariance is positive semidefinite.
g.Covariance(std::move(newCov)); // Set new covariance.

// Compute the probability of the same point [0, 0.5, 0.25].
const double p2 = g.Probability(arma::vec("0 0.5 0.25"));

// Create a Gaussian distribution that is estimated from random samples in 50
// dimensions.
arma::mat samples(50, 10000, arma::fill::randn); // Normally distributed.

mlpack::GaussianDistribution g2(50);
g2.Train(samples);

// Compute the probability of all of the samples.
arma::vec probabilities;
g2.Probability(samples, probabilities);

std::cout << "Average probability is: " << arma::mean(probabilities) << "."
    << std::endl;
```

## Metrics

mlpack includes a number of distance metrics for its distance-based techniques.
These all implement the [same API](../developer/metrics.md), providing one
`Evaluate()` method, and can be used with a variety of different techniques,
including:

<!-- TODO: better names for each link -->

 * [`NeighborSearch`](neighbor_search.md)
 * [`RangeSearch`](range_search.md)
 * [`LMNN`](lmnn.md)
 * [`EMST`](emst.md)
 * [`NCA`](nca.md)
 * [`RANN`](rann.md)
 * [`KMeans`](kmeans.md)

Supported metrics:

 * [`LMetric`](#lmetric): generalized L-metric/Lp-metric, including
   Manhattan/Euclidean/Chebyshev distances
 * [Implement a custom metric](../developer/metrics.md)

### `LMetric`

The `LMetric` template class implements a [generalized
L-metric](https://en.wikipedia.org/wiki/Lp_space#Definition)
(L1-metric, L2-metric, etc.).  The class has two template parameters:

```c++
LMetric<Power, TakeRoot>
```

 * `Power` is an `int` representing the type of the metric; e.g., `2` would
   represent the L2-metric (Euclidean distance).
   - `Power` must be `1` or greater.
   - If `Power` is `INT_MAX`, the metric is the L-infinity distance (Chebyshev
     distance).

 * `TakeRoot` is a `bool` (default `true`) indicating whether the root of the
   distance should be taken.
   - If set to `false`, the metric will no longer satisfy the triangle
     inequality.

---

Several convenient typedefs are available:

 * `ManhattanDistance` (defined as `LMetric<1>`)
 * `EuclideanDistance` (defined as `LMetric<2>`)
 * `SquaredEuclideanDistance` (defined as `LMetric<2, false>`)
 * `ChebyshevDistance` (defined as `LMetric<INT_MAX>`)

---

The static `Evaluate()` method can be used to compute the distance between two
vectors.

*Note:* The vectors given to `Evaluate()` can have any type so long as the type
implements the Armadillo API (e.g. `arma::fvec`, `arma::sp_fvec`, etc.).

---

*Example usage:*

```c++
// Create two vectors: [0, 1.0, 5.0] and [1.0, 3.0, 5.0].
arma::vec a("0.0 1.0 5.0");
arma::vec b("1.0 3.0 5.0");

const double d1 = mlpack::ManhattanDistance::Evaluate(a, b);        // d1 = 3.0
const double d2 = mlpack::EuclideanDistance::Evaluate(a, b);        // d2 = 2.24
const double d3 = mlpack::SquaredEuclideanDistance::Evaluate(a, b); // d3 = 5.0
const double d4 = mlpack::ChebyshevDistance::Evaluate(a, b);        // d4 = 2.0
const double d5 = mlpack::LMetric<4>::Evaluate(a, b);               // d5 = 2.03
const double d6 = mlpack::LMetric<3, false>::Evaluate(a, b);        // d6 = 9.0

std::cout << "Manhattan distance:         " << d1 << "." << std::endl;
std::cout << "Euclidean distance:         " << d2 << "." << std::endl;
std::cout << "Squared Euclidean distance: " << d3 << "." << std::endl;
std::cout << "Chebyshev distance:         " << d4 << "." << std::endl;
std::cout << "L4-distance:                " << d5 << "." << std::endl;
std::cout << "Cubed L3-distance:          " << d6 << "." << std::endl;

// Compute the distance between two random 10-dimensional vectors in a matrix.
arma::mat m(10, 100, arma::fill::randu);

const double d7 = mlpack::EuclideanDistance::Evaluate(m.col(0), m.col(7));

std::cout << std::endl;
std::cout << "Distance between two random vectors: " << d7 << "." << std::endl;
std::cout << std::endl;

// Compute the distance between two 32-bit precision `float` vectors.
arma::fvec fa("0.0 1.0 5.0");
arma::fvec fb("1.0 3.0 5.0");

const double d8 = mlpack::EuclideanDistance::Evaluate(fa, fb); // d8 = 2.236

std::cout << "Euclidean distance (fvec): " << d8 << "." << std::endl;
```

## Kernels

mlpack includes a number of Mercer kernels for its kernel-based techniques.
These all implement the [same API](../developer/kernels.md), providing one
`Evaluate()` method, and can be used with a variety of different techniques,
including:

<!-- TODO: better names for links below -->

 * [`KDE`](kde.md)
 * [`MeanShift`](mean_shift.md)
 * [`KernelPCA`](kernel_pca.md)
 * [`FastMKS`](fastmks.md)
 * [`NystroemMethod`](nystroem_method.md)

Supported kernels:

 * [`GaussianKernel`](#gaussiankernel): standard Gaussian/radial basis
   function/RBF kernel
 * [Implement a custom kernel](../developer/kernels.md)

### `GaussianKernel`

The `GaussianKernel` class implements the standard [Gaussian
kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) (also called
the _radial basis function kernel_ or _RBF kernel_).

The Gaussian kernel is defined as:
`k(x1, x2) = exp(-|| x1 - x2 ||^2 / (2 * bw^2))`
where `bw` is the bandwidth parameter of the kernel.

---

#### Constructors and properties

 * `g = GaussianKernel(bw=1.0)`
   - Create a `GaussianKernel` with the given bandwidth `bw`.

 * `g.Bandwidth()` returns the bandwidth of the kernel as a `double`.
   - To set the bandwidth, use `g.Bandwidth(newBandwidth)`.

---

#### Kernel evaluation

 * `g.Evaluate(x1, x2)`
   - Compute the kernel value between two vectors `x1` and `x2`.
   - `x1` and `x2` should be vector types that implement the Armadillo API
     (e.g., `arma::vec`).

 * `g.Evaluate(distance)`
   - Compute the kernel value between two vectors, given that the distance
     between those two vectors (`distance`) is already known.
   - `distance` should have type `double`.

---

#### Other utilities

 * `g.Gradient(distance)`
   - Compute the (one-dimensional) gradient of the kernel function with respect
     to the distance between two points, evaluated at `distance`.

 * `g.Normalizer(dimensionality)`
   - Return the [normalizing
     constant](https://en.wikipedia.org/wiki/Radial_basis_function_kernel) of
     the Gaussian kernel for points in the given dimensionality as a `double`.

---

*Example usage:*

```c++
// Create a Gaussian kernel with default bandwidth.
mlpack::GaussianKernel g;

// Create a Gaussian kernel with bandwidth 5.0.
mlpack::GaussianKernel g2(5.0);

// Evaluate the kernel value between two 3-dimensional points.
arma::vec x1("0.5 1.0 1.5");
arma::vec x2("1.5 1.0 0.5");
const double k1 = g.Evaluate(x1, x2);
const double k2 = g2.Evaluate(x1, x2);
std::cout << "Kernel values: " << k1 << " (bw=1.0), " << k2 << " (bw=5.0)."
    << std::endl;

// Evaluate the kernel value when the distance between two points is already
// computed.
const double distance = 1.5;
const double k3 = g.Evaluate(distance);

// Change the bandwidth of the kernel to 2.5.
g.Bandwidth(2.5);
const double k4 = g.Evaluate(x1, x2);
std::cout << "Kernel value with bw=2.5: " << k4 << "." << std::endl;

// Evaluate the kernel value between x1 and all points in a random matrix.
arma::mat r(3, 100, arma::fill::randu);
arma::vec kernelValues(100);
for (size_t i = 0; i < r.n_cols; ++i)
  kernelValues[i] = g.Evaluate(x1, r.col(i));
std::cout << "Average kernel value for random points: "
    << arma::mean(kernelValues) << "." << std::endl;

// Compute the kernel value between two 32-bit floating-point vectors.
arma::fvec fx1("0.5 1.0 1.5");
arma::fvec fx2("1.5 1.0 0.5");
const double k5 = g.Evaluate(fx1, fx2);
const double k6 = g2.Evaluate(fx1, fx2);
```
