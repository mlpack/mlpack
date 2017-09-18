/*! @page cv Cross-Validation

@section cvintro Introduction
In this tutorial we will see the usage examples of the cross-validation module.

@section cvbasic Basic Usage

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

@section cvexamples More Examples

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
  // Random responses for every point from the code snippet in the beginning of
  // the tutorial.
  arma::rowvec responses = arma::randu<arma::rowvec>(100);

  SimpleCV<LinearRegression, MSE> cv4(0.2, data, responses);
  double lrLambda = 0.05;
  double lrMSE = cv4.Evaluate(lrLambda);
@endcode

The whole list of constructors for a cross-validation class you can find in the
related header file.

*/
