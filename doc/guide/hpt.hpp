/*! @page hpt Hyper-Parameter Tuning

@section hptintro Introduction
In this tutorial we will see the usage examples of the hyper-parameter tuning
module.

@section hptbasic Basic Usage

The interface of the hyper-parameter tuning module is quite similar to the
interface of the @ref cv module. To construct a \c HyperParameterTuner object
you need to specify what machine learning algorithm, cross-validation strategy,
metric and optimization strategy (\c GridSearch will be used by default) you are
going to use, and then to pass the same arguments as we do for cross-validation
classes. Let's see some examples.

Suppose we have the following data to train and validate on.
@code
  // 100-point 5-dimensional random dataset.
  arma::mat data = arma::randu<arma::mat>(5, 100);
  // Noisy responses retrieved by a random linear transformation of data.
  arma::rowvec responses = arma::randu<arma::rowvec>(5) * data +
      0.1 * arma::randn<arma::rowvec>(100);
@endcode

Then we can use the following code to try to find a good \c lambda value for
\c LinearRegression.

@code
  // Using 80% of data for training and remaining 20% for assessing MSE.
  double validationSize = 0.2;
  HyperParameterTuner<LinearRegression, MSE, SimpleCV> hpt(validationSize,
      data, responses);

  // Finding a good value for lambda from the values 0.0, 0.001, 0.01, 0.1,
  // and 1.0.
  arma::vec lambdas{0.0, 0.001, 0.01, 0.1, 1.0};
  double bestLambda;
  std::tie(bestLambda) = hpt.Optimize(lambdas);
@endcode

In this example we have used GridSearch (the default optimizer) to find a good
value for the \c lambda hyper-parameter. For that we have specified what values
should be tried.

@section hptfixed Fixed Arguments

When some hyper-parameters should not be optimized, you can specify values
for them with the \c Fixed function as in the following example of trying to
find good \c lambda1 and \c lambda2 values for \c LARS.

@code
  HyperParameterTuner<LARS, MSE, SimpleCV> hpt2(validationSize, data,
      responses);

  bool transposeData = true;
  bool useCholesky = false;
  arma::vec lambda1Set{0.0, 0.001, 0.01, 0.1, 1.0};
  arma::vec lambda2Set{0.0, 0.002, 0.02, 0.2, 2.0};

  double bestLambda1, bestLambda2;
  std::tie(bestLambda1, bestLambda2) = hpt2.Optimize(Fixed(transposeData),
      Fixed(useCholesky), lambda1Set, lambda2Set);
@endcode
Note that we have used the same order of arguments as they appear in the \c LARS
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

When we know approximate optimal values (which we can try to find with the
\c GridSearch optimizer) for real-valued hyper-parameters, we can try to tune
them even more with gradient-based optimization. In the following example we
try to optimize the \c lambda1 and \c lambda2 hyper-parameters for \c LARS with
the \c GradientDescent optimizer.
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

*/
