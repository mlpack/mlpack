# Hyper-parameter Tuning

mlpack implements a generic hyperparameter tuner that is able to tune both
continuous and discrete parameters of various different algorithms.  This is an
important task---the performance of many machine learning algorithms can be
highly dependent on the hyperparameters that are chosen for that algorithm.
(One example: the choice of `k` for a `k`-nearest-neighbors classifier.)

This hyper-parameter tuner is built on the same general concept as the
cross-validation classes (see the [cross-validation tutorial](cv.md)): given
some machine learning algorithm, some data, some performance measure, and a set
of hyperparameters, attempt to find the hyperparameter set that best optimizes
the performance measure on the given data with the given algorithm.

mlpack's implementation of hyperparameter tuning is flexible, and is built in a
way that supports many algorithms and many optimizers.  At the time of this
writing, complex hyperparameter optimization techniques are not available, but
the hyperparameter tuner does support these, should they be implemented in the
future.

In this tutorial we will see the usage examples of the hyper-parameter tuning
module, and also more details about the `HyperParameterTuner` class.

## Basic Usage

The interface of the hyper-parameter tuning module is quite similar to the
interface of the [cross-validation module](cv.md). To construct a
`HyperParameterTuner` object you need to specify as template parameters what
machine learning algorithm, cross-validation strategy, performance measure, and
optimization strategy (`ens::GridSearch` will be used by default) you are going
to use.  Then, you must pass the same arguments as for the cross-validation
classes: the data and labels (or responses) to use are given to the constructor,
and the possible hyperparameter values are given to the
`HyperParameterTuner::Optimize()` method, which returns the best algorithm
configuration as a `std::tuple<>`.

Let's see some examples.

Suppose we have the following data to train and validate on.

```c++
// 100-point 5-dimensional random dataset.
arma::mat data = arma::randu<arma::mat>(5, 100);
// Noisy responses retrieved by a random linear transformation of data.
arma::rowvec responses = arma::randu<arma::rowvec>(5) * data +
    0.1 * arma::randn<arma::rowvec>(100);
```

Given the dataset above, we can use the following code to try to find a good
`lambda` value for `LinearRegression`.  Here we use `cv::SimpleCV` instead of
k-fold cross-validation to save computation time.

```c++
// Using 80% of data for training and remaining 20% for assessing MSE.
double validationSize = 0.2;
HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
    data, responses);

// Finding a good value for lambda from the discrete set of values 0.0, 0.001,
// 0.01, 0.1, and 1.0.
arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
double bestLambda;
std::tie(bestLambda) = hpt.Optimize(lambdas);
```

In this example we have used `ens::GridSearch` (the default optimizer) to find a
good value for the `lambda` hyper-parameter.  For that we have specified what
values should be tried.

## Fixed Arguments

When some hyper-parameters should not be optimized, you can specify values for
them with the `Fixed()` method as in the following example of trying to find
good `lambda1` and `lambda2` values for `LARS` (least-angle regression).

```c++
HyperParameterTuner<LARS, MSE, SimpleCV> hpt2(validationSize, data,
    responses);

// The hyper-parameter tuner should not try to change the transposeData or
// useCholesky parameters.
bool transposeData = true;
bool useCholesky = false;

// We wish only to search for the best lambda1 and lambda2 values.
arma::vec lambda1Set{0.0, 0.001, 0.01, 0.1, 1.0};
arma::vec lambda2Set{0.0, 0.002, 0.02, 0.2, 2.0};

double bestLambda1, bestLambda2;
std::tie(bestLambda1, bestLambda2) = hpt2.Optimize(Fixed(transposeData),
    Fixed(useCholesky), lambda1Set, lambda2Set);
```

Note that for the call to `hpt2.Optimize()`, we have used the same order of
arguments as they appear in the corresponding `LARS` constructor:

```c++
LARS(const arma::mat& data,
     const arma::rowvec& responses,
     const bool transposeData = true,
     const bool useCholesky = false,
     const double lambda1 = 0.0,
     const double lambda2 = 0.0,
     const double tolerance = 1e-16);
```

## Gradient-based optimization

In some cases we may wish to optimize a hyperparameter over the space of all
possible real values, instead of providing a grid in which to search.
Alternately, we may know approximately optimal values from a grid search for
real-valued hyperparameters, but wish to further tune those values.

In this case, we can use a gradient-based optimizer for hyperparameter search.
In the following example, we try to optimize the `lambda1` and `lambda2`
hyper-parameters for `LARS` with the `ens::GradientDescent` optimizer.

```c++
HyperParameterTuner<LARS, MSE, SimpleCV, GradientDescent> hpt3(validationSize,
    data, responses);

// GradientDescent can be adjusted in the following way.
hpt3.Optimizer().StepSize() = 0.1;
hpt3.Optimizer().Tolerance() = 1e-15;

// We can set up values used for calculating gradients.
hpt3.RelativeDelta() = 0.01;
hpt3.MinDelta() = 1e-10;

double initialLambda1 = 0.001;
double initialLambda2 = 0.002;

double bestGDLambda1, bestGDLambda2;
std::tie(bestGDLambda1, bestGDLambda2) = hpt3.Optimize(Fixed(transposeData),
    Fixed(useCholesky), initialLambda1, initialLambda2);
```

## The `HyperParameterTuner` class

The `HyperParameterTuner` class is very similar to the `KFoldCV` and `SimpleCV`
classes (see the [cross-validation tutorial](cv.md) for more information on
those two classes), but there are a few important differences.

First, the `HyperParameterTuner` accepts five different hyperparameters; only
the first three of these are required:

  - `MLAlgorithm` This is the algorithm to be used.
  - `Metric` This is the performance measure to be used; see
        [the cross-validation tutorial](cv.md) for more information.
  - `CVType` This is the type of cross-validation to be used for evaluating the
        performance measure; this should be `KFoldCV` or `SimpleCV`.
  - `OptimizerType` This is the type of optimizer to use; it can be
        `GridSearch` or a gradient-based optimizer.
  - `MatType` This is the type of data matrix to use.  The default is
        `arma::mat`.  This only needs to be changed if you are specifically
        using sparse data, or if you want to use a numeric type other than
        `double`.

The last two template parameters are automatically inferred by the
`HyperParameterTuner` and should not need to be manually specified, unless an
unconventional data type like `arma::fmat` is being used for data points.

Typically, `SimpleCV` is a good choice for `CVType` because it takes so much
less time to compute than full `KFoldCV`; however, the disadvantage is that
`SimpleCV` might give a somewhat more noisy estimate of the performance measure
on unseen test data.

The constructor for the `HyperParameterTuner` is called with exactly the same
arguments as the corresponding `CVType` that has been chosen.  For more
information on that, please see the [cross-validation tutorial](cv.md).  As an
example, if we are using `SimpleCV` and wish to hold out 20% of the dataset as a
validation set, we might construct a `HyperParameterTuner` like this:

```c++
// We will use LinearRegression as the MLAlgorithm, and MSE as the performance
// measure.  Our dataset is 'dataset' and the responses are 'responses'.
HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(0.2, dataset,
    responses);
```

Next, we must set up the hyperparameters to be optimized.  If we are doing a
grid search with the \c ens::GridSearch optimizer (the
default), then we only need to pass a `std::vector` (for non-numeric
hyperparameters) or an `arma::vec` (for numeric hyperparameters) containing all
of the possible choices that we wish to search over.

For instance, a set of numeric values might be chosen like this, for the
`lambda` parameter (of type `double`):

```c++
arma::vec lambdaSet = arma::vec("0.0 0.1 0.5 1.0");
```

Similarly, a set of non-numeric values might be chosen like this, for the
`intercept` parameter:

```c++
std::vector<bool> interceptSet = { false, true };
```

Once all of these are set up, the `HyperParameterTuner::Optimize()` method may
be called to find the best set of hyperparameters:

```c++
bool intercept;
double lambda;
std::tie(lambda, intercept) = hpt.Optimize(lambdaSet, interceptSet);
```

Alternately, the `Fixed()` method (detailed in the "Fixed arguments" section)
can be used to fix the values of some parameters.

For continuous optimizers like `ens::GradientDescent`, a range does not need to
be specified but instead only a single value.  See the "Gradient-Based
Optimization" section for more details.

## Further documentation

For more information on the `HyperParameterTuner` class, see the source code fro
the `HyperParameterTuner` class (it is very well commented!), and the
[cross-validation tutorial](cv.md).
