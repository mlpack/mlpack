namespace mlpack {
namespace hpt {

/*! @page hpt Hyper-Parameter Tuning

@section hptintro Introduction

\b mlpack implements a generic hyperparameter tuner that is able to tune both
continuous and discrete parameters of various different algorithms.  This is an
important task---the performance of many machine learning algorithms can be
highly dependent on the hyperparameters that are chosen for that algorithm.
(One example: the choice of \f$k\f$ for a \f$k\f$-nearest-neighbors classifier.)

This hyper-parameter tuner is built on the same general concept as the
cross-validation classes (see the @ref cv "cross-validation tutorial"): given
some machine learning algorithm, some data, some performance measure, and a set
of hyperparameters, attempt to find the hyperparameter set that best optimizes
the performance measure on the given data with the given algorithm.

\b mlpack's implementation of hyperparameter tuning is flexible, and is built in
a way that supports many algorithms and many optimizers.  At the time of this
writing, complex hyperparameter optimization techniques are not available, but
the hyperparameter tuner does support these, should they be implemented in the
future.

In this tutorial we will see the usage examples of the hyper-parameter tuning
module, and also more details about the \c HyperParameterTuner class.

@section hptbasic Basic Usage

The interface of the hyper-parameter tuning module is quite similar to the
interface of the @ref cv "cross-validation module". To construct a \c
HyperParameterTuner object you need to specify as template parameters what
machine learning algorithm, cross-validation strategy, performance measure, and
optimization strategy (\ref optimization::GridSearch "GridSearch" will be used by
default) you are going to use.  Then, you must pass the same arguments as for
the cross-validation classes: the data and labels (or responses) to use are
given to the constructor, and the possible hyperparameter values are given to
the \c HyperParameterTuner::Optimize() method, which returns the best
algorithm configuration as a \c std::tuple<>.

Let's see some examples.

Suppose we have the following data to train and validate on.
@code
  // 100-point 5-dimensional random dataset.
  arma::mat data = arma::randu<arma::mat>(5, 100);
  // Noisy responses retrieved by a random linear transformation of data.
  arma::rowvec responses = arma::randu<arma::rowvec>(5) * data +
      0.1 * arma::randn<arma::rowvec>(100);
@endcode

Given the dataset above, we can use the following code to try to find a good \c
lambda value for \ref regression::LinearRegression "LinearRegression".  Here we
use \ref cv::SimpleCV "SimpleCV" instead of k-fold cross-validation to save
computation time.

@code
  // Using 80% of data for training and remaining 20% for assessing MSE.
  double validationSize = 0.2;
  HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
      data, responses);

  // Finding a good value for lambda from the discrete set of values 0.0, 0.001,
  // 0.01, 0.1, and 1.0.
  arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
  double bestLambda;
  std::tie(bestLambda) = hpt.Optimize(lambdas);
@endcode

In this example we have used \ref optimization::GridSearch "GridSearch" (the
default optimizer) to find a good value for the \c lambda hyper-parameter.  For
that we have specified what values should be tried.

@section hptfixed Fixed Arguments

When some hyper-parameters should not be optimized, you can specify values
for them with the \c Fixed() method as in the following example of trying to
find good \c lambda1 and \c lambda2 values for \ref regression::LARS "LARS"
(least-angle regression).

@code
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
@endcode

Note that for the call to \c hpt2.Optimize(), we have used the same order of
arguments as they appear in the corresponding \ref regression::LARS "LARS"
constructor:

@code
  LARS(const arma::mat& data,
       const arma::rowvec& responses,
       const bool transposeData = true,
       const bool useCholesky = false,
       const double lambda1 = 0.0,
       const double lambda2 = 0.0,
       const double tolerance = 1e-16);
@endcode

@section hptgradient Gradient-Based Optimization

In some cases we may wish to optimize a hyperparameter over the space of all
possible real values, instead of providing a grid in which to search.
Alternately, we may know approximately optimal values from a grid search for
real-valued hyperparameters, but wish to further tune those values.

In this case, we can use a gradient-based optimizer for hyperparameter search.
In the following example, we try to optimize the \c lambda1 and \c lambda2
hyper-parameters for \ref regression::LARS "LARS" with the
\ref optimization::GradientDescent "GradientDescent" optimizer.

@code
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
@endcode

@section hpt_class The HyperParameterTuner class

The \c HyperParameterTuner class is very similar to the
\ref cv::KFoldCV "KFoldCV" and \ref cv::SimpleCV "SimpleCV" classes (see the
@ref "cross-validation tutorial" for more information on those two classes), but
there are a few important differences.

First, the \c HyperParameterTuner accepts five different hyperparameters; only
the first three of these are required:

  - \c MLAlgorithm This is the algorithm to be used.
  - \c Metric This is the performance measure to be used; see
        @ref cvbasic_metrics for more information.
  - \c CVType This is the type of cross-validation to be used for evaluating the
        performance measure; this should be \ref cv::KFoldCV "KFoldCV" or
        \ref cv::SimpleCV "SimpleCV".
  - \c OptimizerType This is the type of optimizer to use; it can be
        \c GridSearch or a gradient-based optimizer.
  - \c MatType This is the type of data matrix to use.  The default is
        \c arma::mat.  This only needs to be changed if you are specifically
        using sparse data, or if you want to use a numeric type other than
        \c double.

The last two template parameters are automatically inferred by the
\c HyperParameterTuner and should not need to be manually specified, unless an
unconventional data type like \c arma::fmat is being used for data points.

Typically, \ref cv::SimpleCV "SimpleCV" is a good choice for \c CVType because
it takes so much less time to compute than full \ref cv::KFoldCV "KFoldCV";
however, the disadvantage is that \ref cv::SimpleCV "SimpleCV" might give a
somewhat more noisy estimate of the performance measure on unseen test data.

The constructor for the \c HyperParameterTuner is called with exactly the same
arguments as the corresponding \c CVType that has been chosen.  For more
information on that, please see the
@ref cvbasic_api "cross-validation constructor tutorial".  As an example, if we
are using \ref cv::SimpleCV "SimpleCV" and wish to hold out 20\% of the dataset
as a validation set, we might construct a \c HyperParameterTuner like this:

@code
// We will use LinearRegression as the MLAlgorithm, and MSE as the performance
// measure.  Our dataset is 'dataset' and the responses are 'responses'.
HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(0.2, dataset,
    responses);
@endcode

Next, we must set up the hyperparameters to be optimized.  If we are doing a
grid search with the \ref optimization::GridSearch "GridSearch" optimizer (the
default), then we only need to pass a `std::vector` (for non-numeric
hyperparameters) or an `arma::vec` (for numeric hyperparameters) containing all
of the possible choices that we wish to search over.

For instance, a set of numeric values might be chosen like this, for the
\c lambda parameter (of type \c double):

@code
arma::vec lambdaSet = arma::vec("0.0 0.1 0.5 1.0");
@endcode

Similarly, a set of non-numeric values might be chosen like this, for the
\c intercept parameter:

@code
std::vector<bool> interceptSet = { false, true };
@endcode

Once all of these are set up, the \c HyperParameterTuner::Optimize() method may
be called to find the best set of hyperparameters:

@code
bool intercept;
double lambda;
std::tie(lambda, intercept) = hpt.Optimize(lambdaSet, interceptSet);
@endcode

Alternately, the \c Fixed() method (detailed in the @ref hptfixed
"Fixed arguments" section) can be used to fix the values of some parameters.

For continuous optimizers like
\ref optimization::GradientDescent "GradientDescent", a range does not need to
be specified but instead only a single value.  See the
\ref hptgradient "Gradient-Based Optimization" section for more details.

@section hptfurther Further documentation

For more information on the \c HyperParameterTuner class, see the
mlpack::hpt::HyperParameterTuner class documentation and the
@ref cv "cross-validation tutorial".

*/

} // namespace hpt
} // namespace mlpack
