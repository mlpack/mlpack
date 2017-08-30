/*! @page cv_and_hpt Cross-Validation and Hyper-Parameter Tuning

@section intro Introduction
In this tutorial we will see the usage examples of the cross-validation and
hyper-parameter tuning modules.

@section cv Cross-Validation

@subsection cv_basic Basic Usage

Suppose we have some data to train and validate on.
@code
  // 100-point 6-dimensional random dataset.
  arma::mat data = arma::randu<arma::mat>(6, 100);
  // Random labels in the [0, 4] interval.
  arma::Row<size_t> labels =
      arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 4));
  size_t numClasses = 5;
@endcode

To run 10-fold cross-validation for softmax regression with accuracy as a
metric we can write the following piece of code.
@code
  KFoldCV<SoftmaxRegression, Accuracy> cv(10, data, labels, numClasses);
  double lambda = 0.1;
  double softmaxAccuracy = cv.Evaluate(lambda);
@endcode
In this example the \c Evaluate method relies on the following \c
SoftmaxRegression constructor:
@code
  template<typename OptimizerType = mlpack::optimization::L_BFGS>
  SoftmaxRegression(const arma::mat& data,
                    const arma::Row<size_t>& labels,
                    const size_t numClasses,
                    const double lambda = 0.0001,
                    const bool fitIntercept = false,
                    OptimizerType optimizer = OptimizerType());
@endcode
which has the parameter \c lambda after three conventional arguments (\c data,
\c labels and \c numClasses). We can skip passing \c fitIntercept and \c
optimizer (as well as \c lambda), since there are the default values.

In general to cross-validate you need to specify what machine learning algorithm
and metric you are going to use, and then to pass some conventional data-related
parameters into one of the cross-validation constructors and all other
parameters (which are hyper-parameters in many cases) into the \c Evaluate
method.

@subsection cv_examples More Examples

In the following example we will cross-validate \c DecisionTree with weights.
@code
  // Random weights for every point from the code snippet above.
  arma::rowvec weights = arma::randu<arma::mat>(1, 100);

  KFoldCV<DecisionTree<>, Accuracy> cv2(10, data, labels, numClasses, weights);
  size_t minimumLeafSize = 8;
  double weightedDecisionTreeAccuracy = cv2.Evaluate(minimumLeafSize);
@endcode
It relies on the following \c DecisionTree constructor:
@code
  template<typename MatType, typename LabelsType, typename WeightsType>
  DecisionTree(MatType&& data,
               LabelsType&& labels,
               const size_t numClasses,
               WeightsType&& weights,
               const size_t minimumLeafSize = 10,
               const std::enable_if_t<arma::is_arma_type<
                   typename std::remove_reference<WeightsType>::type>::value>*
                    = 0);
@endcode
\c DecisionTree models can be constructed in multiple other ways. For example,
if you want to use some particular \c DatasetInfo parameter during construction
of \c DecisionTree objects for cross-validation, you can write the following
code.
@code
  size_t dimensionality = 6;
  data::DatasetInfo datasetInfo(dimensionality);

  KFoldCV<DecisionTree<>, Accuracy> cv3(10, data, datasetInfo, labels,
      numClasses);
  double decisionTreeWithDIAccuracy = cv3.Evaluate(minimumLeafSize);
@endcode
It relies on the following DecisionTree constructor:
@code
  template<typename MatType, typename LabelsType>
  DecisionTree(MatType&& data,
               const data::DatasetInfo& datasetInfo,
               LabelsType&& labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10);
@endcode

\c SimpleCV has the same interface as \c KFoldCV, except it takes as one of its
arguments a proportion (from 0 to 1) of data used as a validation set. For
example, to validate \c LinearRegression with 20\% of training data we can write
the following code.
@code
  // Random responses for every point from the code snippet above.
  arma::rowvec responses = arma::randu<arma::rowvec>(100);

  SimpleCV<LinearRegression, MSE> cv4(0.2, data, responses);
  double lrLambda = 0.05;
  double lrMSE = cv4.Evaluate(lrLambda);
@endcode

The whole list of constructors for a cross-validation class you can find in the
related header file.

@section hpt Hyper-Parameter Tuning

@subsection hpt_basic Basic Usage

The interface of the hyper-parameter tuning module is quite similar to the
interface of the cross-validation module. To construct a \c HyperParameterTuner
object you need to specify what machine learning algorithm, cross-validation
strategy, metric and optimization strategy (\c GridSearch will be used by
default) you are going to use, and then to pass the same arguments as we do for
cross-validation classes. Let's see some examples.

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

@subsection hpt_fixed Fixed Arguments

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

@subsection hpt_gradient Gradient-Based Optimization

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
