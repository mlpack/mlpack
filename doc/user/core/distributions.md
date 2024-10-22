# Distributions

<!-- TODO: link to the completed HMM documentation -->

mlpack has support for a number of different distributions, each supporting the
same API.  These can be used with, for instance, the
[`HMM`](/src/mlpack/methods/hmm/hmm.hpp) class.

 * [`DiscreteDistribution`](#discretedistribution): multidimensional categorical
   distribution (generalized Bernoulli distribution)
 * [`GaussianDistribution`](#gaussiandistribution): multidimensional Gaussian
   distribution
 * [`DiagonalGaussianDistribution`](#diagonalgaussiandistribution):
   multidimensional Gaussian distribution with diagonal covariance
 * [`GammaDistribution`](#gammadistribution): multidimensional Gamma
   distribution, includes exponential, Chi-squared, and Erlang distributions
 * [`LaplaceDistribution`](#laplacedistribution): multidimensional Laplace
   (double exponential) distribution
 * [`RegressionDistribution`](#regressiondistribution): multidimensional
   Gaussian distribution on the errors of a linear regression model

## `DiscreteDistribution`

`DiscreteDistribution` represents a multidimensional categorical distribution
(or generalized Bernoulli distribution) where integer-valued vectors (e.g.
`[0, 3, 4]`) are associated with specific probabilities in each dimension.

*Example:* a 3-dimensional `DiscreteDistribution` will have a specific
probability value associated with each integer value in each dimension.  So, for
the vector `[0, 3, 4]`, `P(0)` in dimension 0 could be, e.g., `0.3`, `P(3)` in
dimension 1 could be, e.g., `0.4`, and `P(4)` in dimension 2 could be, e.g.,
`0.6`.  Then, `P([0, 3, 4])` would be `0.3 * 0.4 * 0.6 = 0.072`.

### Constructors

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

### Access and modify properties of distribution

 * `d.Dimensionality()` returns a `size_t` indicating the number of dimensions
   in the multidimensional discrete distribution.

 * `d.Probabilities(i)` returns an `arma::vec&` containing the probabilities of
   each observation in dimension `i`.
   - `d.Probabilities(i)[j]` is the probability of `j` in dimension `i`.
   - This can be used to modify probabilities: `d.Probabilities(0)[1] = 0.7`
     sets the probability of observing the value `1` in dimension `0` to `0.7`.
   - *Note:* when setting probabilities manually, be sure that the sum of
     probabilities in a dimension is 1!

 * A `DiscreteDistribution` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

### Compute probabilities of points

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

### Sample from the distribution

 * `d.Random()` returns an `arma::vec` with a random sample from the
   multidimensional discrete distribution.

### Fit the distribution to observations

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

### Example usage

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

### Using different element types

The `DiscreteDistribution` class takes two template parameters:

```
DiscreteDistribution<MatType, ObsMatType>
```

 * `MatType` represents the matrix type used to represent internal parameters
   (e.g. probabilities of each observation).
 * `ObsMatType` represents the matrix type used to represent observations.

 * By default:
   - `MatType` is `arma::mat`, but any dense matrix type matching the Armadillo
     API that holds floating-point numbers can be used (e.g. `arma::fmat`).
   - `ObsMatType` is `MatType`, but any matrix type matching the Armadillo API
     can be used (e.g. `arma::fmat`, `arma::imat`, etc.).

 * When using custom `MatType` and `ObsMatType` parameters, several method
   signatures will change:
   - `DiscreteDistributions(probabilities)` will expect `probabilities` to be `
     `std::vector<VecType>`, where `VecType` is the column vector type
     associated with `MatType` (e.g. `arma::fvec` for `arma::fmat`).

   - `Probability(observation)` and `LogProbability(observation)` will expect
     `observation` to be an `ObsVecType`, where `ObsVecType` is the column
     vector type associated with `ObsMatType`, and will return a probability
     with type equivalent to the element type of `MatType`.

   - `Probability(observations, probabilities)` and
     `LogProbability(observations, probabilities)` will expect `observations` to
     be of type `ObsMatType` and `probabilities` to be of type `VecType`.

   - `Random()` will return an `ObsVecType`.

   - `Train(observations)` and `Train(observations, probabilities)` will expect
     `observations` to be of type `ObsMatType` and `probabilities` to be of type
     `VecType`.

   - `Probabilities(dim)` will return a `VecType`.

The code below uses a `DiscreteDistribution` built on 32-bit floating point
numbers.

```c++
// Create a distribution with 10 observations in each of 3 dimensions.
mlpack::DiscreteDistribution<arma::fmat> d(arma::Col<size_t>("10 10 10"));

// Train the distribution on random data.
arma::fmat observations =
    arma::randi<arma::fmat>(3, 100, arma::distr_param(0, 9));
d.Train(observations);

// Compute and print the probability of [8, 6, 7].
const float p = d.Probability(arma::fvec("8 6 7"));
std::cout << "Probability of [8, 6, 7]: " << p << "." << std::endl;
```

The code below uses a `DiscreteDistribution` that internally uses `float` to
hold probabilities, but accepts `unsigned int`s as observations.

```c++
// Create a distribution with 10 observations in each of 3 dimensions.
mlpack::DiscreteDistribution<arma::fmat, arma::umat> d(
    arma::Col<size_t>("10 10 10"));

// Train the distribution on random data.  Note that the observation type is a
// matrix of unsigned ints (arma::umat).
arma::umat observations =
    arma::randi<arma::umat>(3, 100, arma::distr_param(0, 9));
d.Train(observations);

// Compute and print the probability of [8, 6, 7].  Note that the input vector
// is a vector of unsigned ints (arma::uvec), but the returned probability is a
// float because MatType is set to arma::fmat.
const float p = d.Probability(arma::uvec("8 6 7"));
std::cout << "Probability of [8, 6, 7]: " << p << "." << std::endl;

// Print the probability vector for dimension 0.
std::cout << "Probabilities for observations in dimension 0: "
    << d.Probabilities(0).t() << std::endl;
```

## `GaussianDistribution`

`GaussianDistribution` is a standard multivariate Gaussian distribution with
parameterized mean and covariance.  (For a Gaussian distribution with a diagonal
covariance, see
[`DiagonalGaussianDistribution`](#diagonalgaussiandistribution).)

### Constructors

 * `g = GaussianDistribution(dimensionality)`
   - Create the distribution with the given dimensionality.
   - The distribution will have a zero mean and unit diagonal covariance matrix.

 * `g = GaussianDistribution(mean, covariance)`
   - Create the distribution with the given mean and covariance.
   - `mean` is of type `arma::vec` and should have length equal to the
     dimensionality of the distribution.
   - `covariance` is of type `arma::mat`, and should be symmetric and square,
     with rows and columns equal to the dimensionality of the distribution.

### Access and modify properties of distribution

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

 * A `GaussianDistribution` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

### Compute probabilities of points

 * `g.Probability(observation)` returns the probability of the given
   observation as a `double`.
   - `observation` should be an `arma::vec` of size `g.Dimensionality()`.

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

### Sample from the distribution

 * `g.Random()` returns an `arma::vec` with a random sample from the
   multidimensional Gaussian distribution.

### Fit the distribution to observations

 * `g.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `g.Dimensionality()`; `observations.n_cols` is the number of observations.

 * `g.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `g`.

### Example usage

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

### Using different element types

The `GaussianDistribution` class takes one template parameter:

```
GaussianDistribution<MatType>
```

 * `MatType` represents the matrix type used to represent observations.
 * By default, `MatType` is `arma::mat`, but any matrix type matching the
   Armadillo API can be used (e.g. `arma::fmat`).
 * When `MatType` is set to anything other than `arma::mat`, all arguments are
   adapted accordingly:
   - `arma::mat` arguments will instead be `MatType`.
   - `arma::vec` arguments will instead be the corresponding column vector type
     associated with `MatType`.
   - `double` arguments will instead be the element type of `MatType`.

The code below uses a Gaussian distribution to make predictions with 32-bit
floating point numbers.

```c++
// Create a 3-dimensional 32-bit floating point Gaussian distribution with
// random mean and unit covariance.
mlpack::GaussianDistribution<arma::fmat> g(3);
g.Mean().randu();

// Compute the probability of the point [0.2, 0.3, 0.4].
const float p = g.Probability(arma::fvec("0.2 0.3 0.4"));

std::cout << "Probability of (0.2, 0.3, 0.4): " << p << "." << std::endl;
```

## `DiagonalGaussianDistribution`

`DiagonalGaussianDistribution` is a standard multiviate Gaussian distribution
with parameterized mean and diagonal covariance.  (For a full-covariance
Gaussian distribution, see [`GaussianDistribution`](#gaussiandistribution).)

### Constructors

 * `d = DiagonalGaussianDistribution(dimensionality)`
   - Create the distribution with the given dimensionality.
   - The distribution will have a zero mean and unit diagonal covariance matrix.

 * `d = DiagonalGaussianDistribution(mean, covariance)`
   - Create the distribution with the given mean and covariance.
   - `mean` is of type `arma::vec` and should have length equal to the
     dimensionality of the distribution.
   - `covariance` is of type `arma::vec`, and should have length equal to the
     dimensionality of the distribution.  Its elements represent the diagonal of
     the covariance matrix.

### Access and modify properties of distribution

 * `d.Dimensionality()` returns the dimensionality of the distribution as a
   `size_t`.

 * `d.Mean()` returns an `arma::vec&` holding the mean of the distribution.
   This can be modified.

 * `d.Covariance()` returns a `const arma::vec&` holding the covariance of the
   distribution.  To set a new covariance, use `d.Covariance(newCov)` or
   `d.Covariance(std::move(newCov))`, where `newCov` is the new diagonal of the
   covariance matrix.

 * A `DiagonalGaussianDistribution` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

### Compute probabilities of points

 * `d.Probability(observation)` returns the probability of the given observation
   as a `double`.
   - `observation` should be an `arma::vec` of size `d.Dimensionality()`.

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

### Sample from the distribution

 * `d.Random()` returns an `arma::vec` with a random sample from the
   multidimensional diagonal Gaussian distribution.

### Fit the distribution to observations

 * `d.Train(observations)`
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

### Example usage

```c++
// Create a diagonal Gaussian distribution in 3 dimensions with zero mean and
// unit covariance.
mlpack::DiagonalGaussianDistribution d(3);

// Compute the probability of the point [0, 0.5, 0.25].
const double p = d.Probability(arma::vec("0 0.5 0.25"));

// Modify the mean in dimension 0.
d.Mean()[0] = 0.5;

// Set the covariance to a random diagonal.
arma::vec newCovDiag(3, arma::fill::randu);
d.Covariance(std::move(newCovDiag)); // Set new covariance.

// Compute the probability of the same point [0, 0.5, 0.25].
const double p2 = d.Probability(arma::vec("0 0.5 0.25"));

// Create a diagonal Gaussian distribution that is estimated from random samples
// in 50 dimensions.
arma::mat samples(50, 10000, arma::fill::randn); // Normally distributed.

mlpack::DiagonalGaussianDistribution d2(50);
d2.Train(samples);

// Compute the probability of all of the samples.
arma::vec probabilities;
d2.Probability(samples, probabilities);

std::cout << "Average probability is: " << arma::mean(probabilities) << "."
    << std::endl;
```

### Using different element types

The `DiagonalGaussianDistribution` class takes one template parameter:

```
DiagonalGaussianDistribution<MatType>
```

 * `MatType` represents the matrix type used to represent observations.
 * By default, `MatType` is `arma::mat`, but any matrix type matching the
   Armadillo API can be used (e.g. `arma::fmat`).
 * When `MatType` is set to anything other than `arma::mat`, all arguments are
   adapted accordingly:
   - `arma::mat` arguments will instead be `MatType`.
   - `arma::vec` arguments will instead be the corresponding column vector type
     associated with `MatType`.
   - `double` arguments will instead be the element type of `MatType`.

The code below uses a Gaussian distribution to make predictions with 32-bit
floating point numbers.

```c++
// Create a 3-dimensional 32-bit floating point Gaussian distribution with
// random mean and covariance.
mlpack::DiagonalGaussianDistribution<arma::fmat> g(
        arma::randu<arma::fvec>(3), arma::randu<arma::fvec>(3));

// Compute the probability of the point [0.2, 0.3, 0.4].
const float p = g.Probability(arma::fvec("0.2 0.3 0.4"));

std::cout << "Probability of (0.2, 0.3, 0.4): " << p << "." << std::endl;
```

## `GammaDistribution`

`GammaDistribution` is a multivariate Gamma distribution with two parameters for
shape (alpha) and inverse scale (beta).  Certain settings of these parameters
yield the exponential distribution, Chi-squared distribution, and Erlang
distribution.  This family of distributions is commonly used in Bayesian
statistics.  See more on
[Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution).

### Constructors

 * `g = GammaDistribution(dimensionality)`
   - Create the distribution with the given dimensionality.
   - The distribution will have alpha and beta parameters in each dimension set
     to 0.

 * `g = GammaDistribution(alphas, betas)`
   - Create the distribution with the given parameters.
   - `alphas` and `betas` are of type `arma::vec` and should have length equal
     to the dimensionality of the distribution.
   - `alphas` should hold the desired shape parameters in each dimension.
   - `betas` should hold the desired inverse scale parameters in each dimension.

 * `g = GammaDistribution(data, tol=1e-8)`
   - Create the distribution by fitting to the given `data`.
   - `tol` specifies the convergence tolerance for the fitting procedure.
   - Using this constructor is equivalent to calling `g.Train(data, tol)` after
     initializing a `GammaDistribution`.

### Access and modify properties of distribution

 * `g.Dimensionality()` returns the dimensionality of the distribution.

 * `g.Alpha(i)` returns a `double` representing the shape parameter for
   dimension `i`.  `g.Alpha(i) = a` will set the `i`'th dimension's shape
   parameter to `a`.

 * `g.Beta(i)` returns a `double` representing the inverse scale parameter for
   dimension `i`.  `g.Beta(i) = b` will set the `i`'th dimension's inverse scale
   parameter to `b`.

 * A `GammaDistribution` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

### Compute probabilities of points

 * `g.Probability(observation)` returns the probability of the given observation
   as a `double`.
   - `observation` should be an `arma::vec` of size `g.Dimensionality()`.

 * `g.Probability(observations, probabilities)` computes the probabilities of
   many observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `g.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `g.Probability(observations.col(i))`.

 * `g.LogProbability(observation)` returns the log-probability of the given
   observation as a `double`.

 * `g.LogProbability(observations, probabilities)` computes the
   log-probabilities of many observations.

### Sample points from the distribution

 * `g.Random()` returns an `arma::vec` with a random sample from the
   Gamma distribution.

### Fit the distribution to observations

 * `g.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `g.Dimensionality()`; `observations.n_cols` is the number of observations.

 * `g.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `g`.

 * The algorithm used for fitting the distribution is described in the paper
   [Estimating a Gamma Distribution](https://tminka.github.io/papers/minka-gamma.pdf).

### Example usage

```c++
// Create a Gamma distribution in 3 dimensions with ones for the alpha (shape)
// parameters and random beta (inverse scale) parameters.
mlpack::GammaDistribution g(arma::ones<arma::vec>(3) /* shape */,
                            arma::randu<arma::vec>(3) /* scale */);

// Compute the probability and log-probability of the point [0, 0.5, 0.25].
const double p = g.Probability(arma::vec("0 0.5 0.25"));
const double lp = g.LogProbability(arma::vec("0 0.5 0.25"));

std::cout << "Probability of [0 0.5 0.25]:     " << p << "." << std::endl;
std::cout << "Log-probability of [0 0.5 0.25]: " << lp << "." << std::endl;

// Modify the scale and inverse shape parameters in dimension 0.
g.Alpha(0) = 0.5;
g.Beta(0) = 3.0;

// Compute the probability of the same point [0, 0.5, 0.25].
const double p2 = g.Probability(arma::vec("0 0.5 0.25"));
const double lp2 = g.LogProbability(arma::vec("0 0.5 0.25"));

std::cout << "After parameter changes:" << std::endl;
std::cout << "Probability of [0 0.5 0.25]:     " << p << "." << std::endl;
std::cout << "Log-probability of [0 0.5 0.25]: " << lp << "." << std::endl;

// Create a Gamma distribution that is estimated from random samples in 5
// dimensions.  Note that the samples here are uniformly distributed---so a
// Gamma distribution fit will not be a good one!
arma::mat samples(5, 1000, arma::fill::randu);
samples += 2.0; // Shift samples away from zero.

mlpack::GammaDistribution g2(samples, 1e-3 /* tolerance for fitting */);

// Compute the probability of all of the samples.
arma::vec probabilities;
g2.Probability(samples, probabilities);

std::cout << "Average probability is: " << arma::mean(probabilities) << "."
    << std::endl;
```

### Using different element types

The `GammaDistribution` class takes one template parameter:

```
GammaDistribution<MatType>
```

 * `MatType` represents the matrix type used to represent observations.
 * By default, `MatType` is `arma::mat`, but any matrix type matching the
   Armadillo API can be used (e.g. `arma::fmat`).
 * When `MatType` is set to anything other than `arma::mat`, all arguments are
   adapted accordingly:
   - `arma::mat` arguments will instead be `MatType`
   - `arma::vec` arguments will instead be the corresponding column vector type
     associated with `MatType`
   - `double` arguments will instead be the element type of `MatType`
   - If the element type is `float`, the default tolerance (`tol`) for `Train()`
     is `1e-4`

The code below uses a Gamma distribution to make predictions with 32-bit
floating point numbers.

```c++
// Create a 3-dimensional 32-bit floating point Laplace distribution with
// ones for the shape parameter and random scale parameters.
mlpack::GammaDistribution<arma::fmat> g(arma::ones<arma::fvec>(3) /* shape */,
                                        arma::randu<arma::fvec>(3) /* scale */);

// Compute the probability of the point [0.2, 0.3, 0.4].
const float p = g.Probability(arma::fvec("0.2 0.3 0.4"));

std::cout << "Probability of (0.2, 0.3, 0.4): " << p << "." << std::endl;
```

## `LaplaceDistribution`

`LaplaceDistribution` is a multivariate Laplace distribution parameterized by a
mean vector and a single scale value.  The Laplace distribution is sometimes
also called the *double exponential distribution*.  See more on
[Wikipedia](https://en.wikipedia.org/wiki/Laplace_distribution).

### Constructors

 * `l = LaplaceDistribution(dimensionality, scale=1.0)`
   - Create the distribution with the given dimensionality.
   - The distribution will have mean zero and the given `scale`.
   - `scale` must be greater than 0.

 * `l = LaplaceDistribution(mean, scale)`
   - Create the distribution with the given parameters.
   - `mean` is of type `arma::vec` and should have length equal to the
     dimensionality of the distribution.
   - `scale` must be greater than 0.

### Access and modify properties of distribution

 * `l.Dimensionality()` returns the dimensionality of the distribution.

 * `l.Mean()` returns an `arma::vec&` holding the mean of the distribution.
   This can be modified.

 * `l.Scale()` returns a `double` representing the distribution's scale
   parameter.  `l.Scale() = s` will set the scale parameter to `s`.

 * A `LaplaceDistribution` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

### Compute probabilities of points

 * `l.Probability(observation)` returns the probability of the given observation
   as a `double`.
   - `observation` should be an `arma::vec` of size `l.Dimensionality()`.

 * `l.Probability(observations, probabilities)` computes the probabilities of
   many observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `l.Dimensionality()`; `observations.n_cols` is the number of observations.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `l.Probability(observations.col(i))`.

 * `l.LogProbability(observation)` returns the log-probability of the given
   observation as a `double`.

 * `l.LogProbability(observations, probabilities)` computes the
   log-probabilities of many observations.

### Sample points from the distribution

 * `l.Random()` returns an `arma::vec` with a random sample from the
   Laplace distribution.

### Fit the distribution to observations

 * `l.Train(observations)`
   - Fit the distribution to the given observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `l.Dimensionality()`; `observations.n_cols` is the number of observations.

 * `l.Train(observations, observationProbabilities)`
   - Fit the distribution to the given observations, as above, but also provide
     probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::vec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that
     `observations.col(i)` is from `l`.

### Example usage

```c++
// Create a Laplace distribution in 3 dimensions with uniform random mean and
// scale parameter 1.
mlpack::LaplaceDistribution l(arma::randu<arma::vec>(3) /* mean */,
                              1.0 /* scale */);

// Compute the probability and log-probability of the point [0, 0.5, 0.25].
const double p = l.Probability(arma::vec("0 0.5 0.25"));
const double lp = l.LogProbability(arma::vec("0 0.5 0.25"));

std::cout << "Probability of [0 0.5 0.25]:     " << p << "." << std::endl;
std::cout << "Log-probability of [0 0.5 0.25]: " << lp << "." << std::endl;

// Modify the scale, and the mean in dimension 1.
l.Scale() = 2.0;
l.Mean()[1] = 1.5;

// Compute the probability of the same point [0, 0.5, 0.25].
const double p2 = l.Probability(arma::vec("0 0.5 0.25"));
const double lp2 = l.LogProbability(arma::vec("0 0.5 0.25"));

std::cout << "After parameter changes:" << std::endl;
std::cout << "Probability of [0 0.5 0.25]:     " << p << "." << std::endl;
std::cout << "Log-probability of [0 0.5 0.25]: " << lp << "." << std::endl;

// Create a Laplace distribution that is estimated from random samples in 50
// dimensions.  Note that the samples here are normally distributed---so a Gamma
// distribution fit will not be a good one!
arma::mat samples(50, 10000, arma::fill::randn);

mlpack::LaplaceDistribution l2;
l2.Train(samples);

// Compute the probability of all of the samples.
arma::vec probabilities;
l2.Probability(samples, probabilities);

std::cout << "Average probability is: " << arma::mean(probabilities) << "."
    << std::endl;
```

### Using different element types

The `LaplaceDistribution` class takes one template parameter:

```
LaplaceDistribution<MatType>
```

 * `MatType` represents the matrix type used to represent observations.
 * By default, `MatType` is `arma::mat`, but any matrix type matching the
   Armadillo API can be used (e.g. `arma::fmat`).
 * When `MatType` is set to anything other than `arma::mat`, all arguments are
   adapted accordingly:
   - `arma::mat` arguments will instead be `MatType`.
   - `arma::vec` arguments will instead be the corresponding column vector type
     associated with `MatType`.
   - `double` arguments will instead be the element type of `MatType`.

The code below uses a Laplace distribution to make predictions with 32-bit
floating point numbers.

```c++
// Create a 3-dimensional 32-bit floating point Laplace distribution with
// random mean and scale of 2.0.
mlpack::LaplaceDistribution<arma::fmat> g(arma::randu<arma::fvec>(3), 2.0);

// Compute the probability of the point [0.2, 0.3, 0.4].
const float p = g.Probability(arma::fvec("0.2 0.3 0.4"));

std::cout << "Probability of (0.2, 0.3, 0.4): " << p << "." << std::endl;
```

## `RegressionDistribution`

The `RegressionDistribution` is a [Gaussian distribution](#gaussiandistribution)
fitted on the errors of a linear regression model.  Given a point `x` with
response `y`, the probability of `(y, x)` is computed using a univariate
Gaussian distribution on the scalar residual `y - y'`, where `y'` is the linear
regression model's prediction on `x`.

<!-- TODO: fix link! -->

This class is meant to be used with mlpack's
[HMM](/src/mlpack/methods/hmm/hmm.hpp) class for the task of
[HMM regression (pdf)](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=93a56eb64e77ac83404fddfd0036e95a742fcee6).

### Constructors

 * `r = RegressionDistribution()`
   - Create an empty `RegressionDistribution`.
   - The distribution will not provide useful predictions; call `Train()` before
     doing anything else with the object!

 * `r = RegressionDistribution(predictors, responses)`
   - Create the `RegressionDistribution` by estimating the parameters with the
     given labeled regression data `predictors` and `responses`.
   - `predictors` should be a
     [column-major](../matrices.md#representing-data-in-mlpack) `arma::mat`
     representing the data the distribution should be trained on.
   - `responses` should be an `arma::rowvec` representing the responses for each
     data point.
   - The number of elements in `responses` (e.g. `responses.n_elem`) should be
     the same as the number of columns in `predictors` (e.g.
     `predictors.n_cols`).

### Access and modify properties of distribution

 * `r.Dimensionality()` returns the dimensionality of the distribution.
   - ***Note***: this is *not* the same as the number of elements in a vector
     passed to `Probability()`!

 * `r.Rf()` returns the [`LinearRegression&`](../methods/linear_regression.md)
   model.  This can be modified.

 * `r.Parameters()` returns an `const arma::vec&` with length
   `r.Dimensionality() + 1` representing the parameters of the linear regression
   model.  The first element is the bias; subsequent elements are the weights
   for each dimension.

 * `r.Err()` returns a [`GaussianDistribution&`](#gaussiandistribution) object
   representing the univariate distribution trained on the model's residuals.
   This can be modified.

 * A `RegressionDistribution` can be serialized with
   [`data::Save()` and `data::Load()`](../load_save.md#mlpack-objects).

### Compute probabilities of points

 * `r.Probability(observation)` returns the probability of the given *labeled*
   observation as a `double`.
   - `observation` should be an `arma::vec` of size `r.Dimensionality() + 1`,
     containing both the data point and its scalar response.
   - The first element of `observation` should be the response; subsequent
     elements should be the data point.

 * `r.Probability(observations, probabilities)` computes the probabilities of
   many labeled observations.
   - `observations` should be an `arma::mat` with number of rows equal to
     `r.Dimensionality() + 1`; `observations.n_cols` is the number of
     observations.
   - The first row of `observations` should correspond to the responses for each
     data point.
   - `probabilities` will be set to size `observations.n_cols`.
   - `probabilities[i]` will be set to `r.Probability(observations.col(i))`.

 * `r.LogProbability(observation)` returns the log-probability of the given
   labeled observation as a `double`.

 * `r.LogProbability(observations, probabilities)` computes the
   log-probabilities of many labeled observations.

### Fit the distribution to observations

Training a `RegressionDistribution` on a given set of labeled observations is
done by first training a [`LinearRegression`](../methods/linear_regression.md)
model on the dataset, and then subsequently training a univariate
[`GaussianDistribution`](#gaussiandistribution) on the residual error of each
data point.

In the `Train()` overloads, the `observations` matrix is expected to contain
both the responses and the data points (predictors).

 * `r.Train(observations)`
   - Fit the distribution to the given *labeled* observations.
   - `observations` should be an `arma::mat` with number of rows equal to the
     dimensionality of the data plus one; `observations.n_cols` is the number of
     observations.
   - The first row of `observations` should correspond to the responses of the
     data; subsequent rows correspond to the data itself.

 * `r.Train(observations, observationProbabilities)`
   - Fit the distribution to the given *labeled* observations, as above, but
     also provide probabilities that each observation is from this distribution.
   - `observationProbabilities` should be an `arma::rowvec` of length
     `observations.n_cols`.
   - `observationProbabilities[i]` should be equal to the probability that the
     `i`'th observation is from `r`.

***Note***: if the linear regression model is able to exactly fit the
observations, then the resulting Gaussian distribution will have zero-valued
standard deviation, and `Probability()` will return `1` for points that are
perfectly fit and `0` otherwise.

### Example usage

```c++
// Create an example dataset that arises from a noisy random linear model:
//
//   y = bx + noise
//
// Noise is added from a Gaussian distribution with zero mean and unit variance.
// Data is 10-dimensional, and we will generate 1000 points.
arma::vec b(10, arma::fill::randu);
arma::mat x(10, 1000, arma::fill::randu);

arma::rowvec y = b.t() * x + arma::randn<arma::rowvec>(1000);

// Now fit a RegressionDistribution to the data.
mlpack::RegressionDistribution r(x, y);

// Print information about the distribution.
std::cout << "RegressionDistribution model parameters:" << std::endl;
std::cout << " - " << r.Parameters().subvec(1, r.Parameters().n_elem - 1).t();
std::cout << " - Bias: " << r.Parameters()[0] << "." << std::endl;
std::cout << "True model parameters:" << std::endl;
std::cout << " - " << b.t();
std::cout << "Error Gaussian mean is " << r.Err().Mean()[0] << ", with "
    << "variance " << r.Err().Covariance()[0] << "." << std::endl << std::endl;

// Compute the probability of a point in the training set.  We must assemble the
// points into a single vector.
arma::vec p1(11); // p1 will be point 5 from (x, y).
p1[0] = y[5];
p1.subvec(1, p1.n_elem - 1) = x.col(5);
std::cout << "Probability of point 5:      " << r.Probability(p1) << "."
    << std::endl;

arma::vec p2(11, arma::fill::randu);
std::cout << "Probability of random point: " << r.Probability(p2) << "."
    << std::endl;

// Print log-probabilities too.
std::cout << "Log-probability of point 5:      " << r.LogProbability(p1) << "."
    << std::endl;
std::cout << "Log-probability of random point: " << r.LogProbability(p2) << "."
    << std::endl << std::endl;

// Change the error distribution.
y = b.t() * x + (1.5 * arma::randn<arma::rowvec>(1000));

// Combine x and y to build the observations matrix for Train().
arma::mat observations(x.n_rows + 1, x.n_cols);
observations.row(0) = y;
observations.rows(1, observations.n_rows - 1) = x;

// Assign a random probability for each point.
arma::rowvec observationProbabilities(observations.n_cols, arma::fill::randu);

// Refit the distribution to the new data.
r.Train(observations, observationProbabilities);

// Print new error distribution information.
std::cout << "Updated error Gaussian mean is " << r.Err().Mean()[0] << ", with "
    << "variance " << r.Err().Covariance()[0] << "." << std::endl << std::endl;

// Compute average probability of points in the dataset.
arma::vec probabilities;
r.Probability(observations, probabilities);
std::cout << "Average probability of points in `observations`: "
    << arma::mean(probabilities) << "." << std::endl;
```

### Using different element types

The `RegressionDistribution` class takes one template parameter:

```
RegressionDistribution<MatType>
```

 * `MatType` represents the matrix type used to represent observations.
 * By default, `MatType` is `arma::mat`, but any matrix type matching the
   Armadillo API can be used (e.g. `arma::fmat`).
 * When `MatType` is set to anything other than `arma::mat`, all arguments are
   adapted accordingly:
   - `arma::mat` arguments will instead be `MatType`.
   - `arma::vec` arguments will instead be the corresponding column vector type
     associated with `MatType`.
   - `double` arguments will instead be the element type of `MatType`.

The code below uses a regression distribution trained on 32-bit floating point
data.

```c++
// Create an example dataset that arises from a noisy random linear model:
//
//   y = bx + noise
//
// Noise is added from a Gaussian distribution with zero mean and unit variance.
// Data is 3-dimensional, and we will generate 1000 points.
arma::fvec b(3, arma::fill::randu);
arma::fmat x(3, 1000, arma::fill::randu);

arma::frowvec y = b.t() * x + arma::randn<arma::frowvec>(1000);

// Now fit a RegressionDistribution to the data.
mlpack::RegressionDistribution<arma::fmat> r(x, y);

// Compute the probability of the point [0.5, 0.2, 0.3, 0.4].
// (Here 0.5 is the response, and [0.2, 0.3, 0.4] is the point.)
const float p = r.Probability(arma::fvec("0.5 0.2 0.3 0.4"));
std::cout << "Probability of (0.5, 0.2, 0.3, 0.4): " << p << "." << std::endl;
```
