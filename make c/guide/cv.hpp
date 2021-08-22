namespace mlpack {
namespace cv {

/*! @page cv Cross-Validation

@section cvintro Introduction

@b mlpack implements cross-validation support for its learning algorithms, for a
variety of performance measures.  Cross-validation is useful for determining an
estimate of how well the learner will generalize to un-seen test data.  It is a
commonly used part of the data science pipeline.

In short, given some learner and some performance measure, we wish to get an
average of the performance measure given different splits of the dataset into
training data and validation data.  The learner is trained on the training data,
and the performance measure is evaluated on the validation data.

mlpack currently implements two easy-to-use forms of cross-validation:

 - @b simple @b cross-validation, where we simply desire the performance measure
   on a single split of the data into a training set and validation set

 - @b k-fold @b cross-validation, where we split the data k ways and desire the
   average performance measure on each of the k splits of the data

In this tutorial we will see the usage examples and details of the
cross-validation module.  Because the cross-validation code is generic and can
be used with any learner and performance measure, any use of the
cross-validation code in mlpack has to be in C++.

This tutorial is split into the following sections:

 - @ref cvbasic Simple cross-validation examples
   - @ref cvbasic_ex_1 10-fold cross-validation on softmax regression
   - @ref cvbasic_ex_2 10-fold cross-validation on weighted decision trees
   - @ref cvbasic_ex_3 10-fold cross-validation with categorical decision trees
   - @ref cvbasic_ex_4 Simple cross-validation for linear regression
 - @ref cvbasic_metrics Performance measures
 - @ref cvbasic_api The \c KFoldCV and \c SimpleCV classes
 - @ref cvbasic_further Further reference

@section cvbasic Simple cross-validation examples

@subsection cvbasic_ex_1 10-fold cross-validation on softmax regression

Suppose we have some data to train and validate on, as defined below:

@code
  // 100-point 6-dimensional random dataset.
  arma::mat data = arma::randu<arma::mat>(6, 100);
  // Random labels in the [0, 4] interval.
  arma::Row<size_t> labels =
      arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 4));
  size_t numClasses = 5;
@endcode

The code above generates an 100-point random 6-dimensional dataset with 5
classes.

To run 10-fold cross-validation for softmax regression with accuracy as a
performance measure, we can write the following piece of code.

@code
  KFoldCV<SoftmaxRegression, Accuracy> cv(10, data, labels, numClasses);
  double lambda = 0.1;
  double softmaxAccuracy = cv.Evaluate(lambda);
@endcode

Note that the \c Evaluate method of \c KFoldCV takes any hyperparameters of an
algorithm---that is, anything that is not \c data, \c labels, \c numClasses,
\c datasetInfo, or \c weights (those last three may not be present for every
algorithm type).  To be more specific, in this example the \c Evaluate method
relies on the following \ref regression::SoftmaxRegression "SoftmaxRegression"
constructor:

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
optimizer since there are the default values.  (Technically, we don't even need
to pass \c lambda since there is a default value.)

In general to cross-validate you need to specify what machine learning algorithm
and metric you are going to use, and then to pass some conventional data-related
parameters into one of the cross-validation constructors and all other
parameters (which are generally hyperparameters) into the \c Evaluate method.

@subsection cvbasic_ex_2 10-fold cross-validation on weighted decision trees

In the following example we will cross-validate
\ref tree::DecisionTree "DecisionTree" with weights.  This is very similar to
the previous example, except that we also have instance weights for each point
in the dataset.  We can generate weights for the dataset from the previous
example with the code below:

@code
  // Random weights for every point from the code snippet above.
  arma::rowvec weights = arma::randu<arma::mat>(1, 100);
@endcode

Given those weights for each point, we can now perform cross-validation by also
passing the weights to the constructor of \c KFoldCV:

@code
  KFoldCV<DecisionTree<>, Accuracy> cv2(10, data, labels, numClasses, weights);
  size_t minimumLeafSize = 8;
  double weightedDecisionTreeAccuracy = cv2.Evaluate(minimumLeafSize);
@endcode

As with the previous example, internally this call to \c cv2.Evaluate() relies
on the following \ref tree::DecisionTree "DecisionTree" constructor:

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

@subsection cvbasic_ex_3 10-fold cross-validation with categorical decision trees

\ref tree::DecisionTree "DecisionTree" models can be constructed in multiple
other ways. For example, if we have a dataset with both categorical and
numerical features, we can also perform cross-validation by using the associated
\c data::DatasetInfo object.  Thus, given some \c data::DatasetInfo object
called \c datasetInfo (that perhaps was produced by a call to \c data::Load() ),
we can perform k-fold cross-validation in a similar manner to the other
examples:

@code
  KFoldCV<DecisionTree<>, Accuracy> cv3(10, data, datasetInfo, labels,
      numClasses);
  double decisionTreeWithDIAccuracy = cv3.Evaluate(minimumLeafSize);
@endcode

This particular call to \c cv3.Evaluate() relies on the following
\ref tree::DecisionTree "DecisionTree" constructor:

@code
  template<typename MatType, typename LabelsType>
  DecisionTree(MatType&& data,
               const data::DatasetInfo& datasetInfo,
               LabelsType&& labels,
               const size_t numClasses,
               const size_t minimumLeafSize = 10);
@endcode

@subsection cvbasic_ex_4 Simple cross-validation for linear regression

\c SimpleCV has the same interface as \c KFoldCV, except it takes as one of its
arguments a proportion (from 0 to 1) of data used as a validation set. For
example, to validate \ref regression::LinearRegression "LinearRegression" with
20\% of the data used in the validation set we can write the following code.

@code
  // Random responses for every point from the code snippet in the beginning of
  // the tutorial.
  arma::rowvec responses = arma::randu<arma::rowvec>(100);

  SimpleCV<LinearRegression, MSE> cv4(0.2, data, responses);
  double lrLambda = 0.05;
  double lrMSE = cv4.Evaluate(lrLambda);
@endcode

@section cvbasic_metrics Performance measures

The cross-validation classes require a performance measure to be specified.
\b mlpack has a number of performance measures implemented; below is a list:

 - mlpack::cv::Accuracy: a simple measure of accuracy
 - mlpack::cv::F1: the F1 score; depends on an averaging strategy
 - mlpack::cv::MSE: minimum squared error (for regression problems)
 - mlpack::cv::Precision: the precision, for classification problems
 - mlpack::cv::Recall: the recall, for classification problems

In addition, it is not difficult to implement a custom performance measure.  A
class following the structure below can be used:

@code
class CustomMeasure
{
  //
  // This evaluates the metric given a trained model and a set of data (with
  // labels or responses) to evaluate on.  The data parameter will be a type of
  // Armadillo matrix, and the labels will be the labels that go with the model.
  //
  // If you know that your model is a classification model (and thus that
  // ResponsesType will be arma::Row<size_t>), it is ok to replace the
  // ResponsesType template parameter with arma::Row<size_t>.
  //
  template<typename MLAlgorithm, typename DataType, typename ResponsesType>
  static double Evaluate(MLAlgorithm& model,
                         const DataType& data,
                         const ResponsesType& labels)
  {
    // Inside the method you should call model.Predict() and compare the
    // values with the labels, in order to get the desired performance measure
    // and return it.
  }
};
@endcode

Once this is implemented, then \c CustomMeasure (or whatever the class is
called) is easy to use as a custom performance measure with \c KFoldCV or
\c SimpleCV.

@section cvbasic_api The KFoldCV and SimpleCV classes

This section provides details about the \c KFoldCV and \c SimpleCV classes.
The cross-validation infrastructure is based on heavy amounts of template
metaprogramming, so that any \b mlpack learner and any performance measure can
be used.  Both classes have two required template parameters and one optional
parameter:

 - \c MLAlgorithm: the type of learner to be used
 - \c Metric: the performance measure to be evaluated
 - \c MatType: the type of matrix used to store the data

In addition, there are two more template parameters, but these are automatically
extracted from the given \c MLAlgorithm class, and users should not need to
specify these parameters except when using an unconventional type like
\c arma::fmat for data points.

The general structure of the \c KFoldCV and \c SimpleCV classes is split into
two parts:

 - The constructor: create the object, and store the data for the \c MLAlgorithm
        training.
 - The \c Evaluate() method: take any non-data parameters for the
        \c MLAlgorithm and calculate the desired performance measure.

This split is important because it defines the API: all data-related parameters
are passed to the constructor, whereas algorithm hyperparameters are passed to
the \c Evaluate() method.

@subsection cvbasic_api_constructor The KFoldCV and SimpleCV constructors

There are six constructors available for \c KFoldCV and \c SimpleCV, each
tailored for a different learning situation.  Each is given below for the
\c KFoldCV class, but the same constructors are also available for the
\c SimpleCV class, with the exception that instead of specifying \c k, the
number of folds, the \c SimpleCV class takes a parameter between 0 and 1
specifying the percentage of the dataset to use as a validation set.

 - `KFoldCV(k, xs, ys)`: this is for unweighted regression applications and
        two-class classification applications; \c xs is the dataset and \c ys
        are the responses or labels for each point in the dataset.

 - `KFoldCV(k, xs, ys, numClasses)`: this is for unweighted classification
        applications; \c xs is the dataset, \c ys are the class labels for each
        data point, and \c numClasses is the number of classes in the dataset.

 - `KFoldCV(k, xs, datasetInfo, ys, numClasses)`: this is for unweighted
        categorical/numeric classification applications; \c xs is the dataset,
        \c datasetInfo is a data::DatasetInfo object that holds the types of
        each dimension in the dataset, \c ys are the class labels for each data
        point, and \c numClasses is the number of classes in the dataset.

 - `KFoldCV(k, xs, ys, weights)`: this is for weighted regression or
        two-class classification applications; \c xs is the dataset, \c ys are
        the responses or labels for each point in the dataset, and \c weights
        are the weights for each point in the dataset.

 - `KFoldCV(k, xs, ys, numClasses, weights)`: this is for weighted
        classification applications; \c xs is the dataset, \c ys are the class
        labels for each point in the dataset; \c numClasses is the number of
        classes in the dataset, and \c weights holds the weights for each point
        in the dataset.

 - `KFoldCV(k, xs, datasetInfo, ys, numClasses, weights)`: this is for
        weighted cateogrical/numeric classification applications; \c xs is the
        dataset, \c datasetInfo is a data::DatasetInfo object that holds the
        types of each dimension in the dataset, \c ys are the class labels for
        each data point, \c numClasses is the number of classes in each dataset,
        and \c weights holds the weights for each point in the dataset.

Note that the constructor you should use is the constructor that most closely
matches the constructor of the machine learning algorithm you would like
performance measures of.  So, for instance, if you are doing multi-class softmax
regression, you could call the constructor
\c "SoftmaxRegression(xs, ys, numClasses)".  Therefore, for \c KFoldCV you would
call the constructor \c "KFoldCV(k, xs, ys, numClasses)" and for \c SimpleCV you
would call the constructor \c "SimpleCV(pct, xs, ys, numClasses)".

@subsection cvbasic_api_evaluate The Evaluate() method

The other method that \c KFoldCV and \c SimpleCV have is the method to
actually calculate the performance measure: \c Evaluate().  The \c Evaluate()
method takes any hyperparameters that would follow the data arguments to the
constructor or \c Train() method of the given \c MLAlgorithm.  The
\c Evaluate() method takes no more arguments than that, and returns the
desired performance measure on the dataset.

Therefore, let us suppose that we are interested in cross-validating the
performance of a softmax regression model, and that we have constructed
the appropriate \c KFoldCV object using the code below:

@code
KFoldCV<SoftmaxRegression, Precision> cv(k, data, labels, numClasses);
@endcode

The \ref regression::SoftmaxRegression "SoftmaxRegression" class has the
constructor

@code
  template<typename OptimizerType = mlpack::optimization::L_BFGS>
  SoftmaxRegression(const arma::mat& data,
                    const arma::Row<size_t>& labels,
                    const size_t numClasses,
                    const double lambda = 0.0001,
                    const bool fitIntercept = false,
                    OptimizerType optimizer = OptimizerType());
@endcode

Note that all parameters after are \c numClasses are optional.  This means that
we can specify none or any of them in our call to \c Evaluate().  Below is some
example code showing three different ways we can call \c Evaluate() with the
\c cv object from the code snippet above.

@code
// First, call with all defaults.
double result1 = cv.Evaluate();

// Next, call with lambda set to 0.1 and fitIntercept set to true.
double result2 = cv.Evaluate(0.1, true);

// Lastly, create a custom optimizer to use for optimization, and use a lambda
// value of 0.5 and fit no intercept.
optimization::SGD<> sgd(0.05, 50000); // Step size of 0.05, 50k max iterations.
double result3 = cv.Evaluate(0.5, false, sgd);
@endcode

The same general idea applies to any \c MLAlgorithm: all hyperparameters must be
passed to the \c Evaluate() method of \c KFoldCV or \c SimpleCV.

@section cvbasic_further Further references

For further documentation, please see the associated Doxygen documentation for
each of the relevant classes:

 - mlpack::cv::SimpleCV
 - mlpack::cv::KFoldCV
 - mlpack::cv::Accuracy
 - mlpack::cv::F1
 - mlpack::cv::MSE
 - mlpack::cv::Precision
 - mlpack::cv::Recall

If you are interested in implementing a different cross-validation strategy than
k-fold cross-validation or simple cross-validation, take a look at the
implementations of each of those classes to guide your implementation.

In addition, the @ref hpt "hyperparameter tuner" documentation may also be
relevant.

*/

} // namespace cv
} // namespace mlpack
