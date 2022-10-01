# Cross-Validation

mlpack implements cross-validation support for its learning algorithms, for a
variety of performance measures.  Cross-validation is useful for determining an
estimate of how well the learner will generalize to un-seen test data.  It is a
commonly used part of the data science pipeline.

In short, given some learner and some performance measure, we wish to get an
average of the performance measure given different splits of the dataset into
training data and validation data.  The learner is trained on the training data,
and the performance measure is evaluated on the validation data.

mlpack currently implements two easy-to-use forms of cross-validation:

 - *simple* cross-validation, where we simply desire the performance measure
   on a single split of the data into a training set and validation set

 - *k-fold* cross-validation, where we split the data `k` ways and desire the
   average performance measure on each of the `k` splits of the data

In this tutorial we will see the usage examples and details of the
cross-validation module.  Because the cross-validation code is generic and can
be used with any learner and performance measure, any use of the
cross-validation code in mlpack has to be in C++.

## Simple cross-validation examples

This section contains examples, in C++, showing the usage of mlpack's simple
cross-validation functionality.

### 10-fold cross-validation on softmax regression

Suppose we have some data to train and validate on, as defined below:

```c++
// 100-point 6-dimensional random dataset.
arma::mat data = arma::randu<arma::mat>(6, 100);
// Random labels in the [0, 4] interval.
arma::Row<size_t> labels =
    arma::randi<arma::Row<size_t>>(100, arma::distr_param(0, 4));
size_t numClasses = 5;
```

The code above generates an 100-point random 6-dimensional dataset with 5
classes.

To run 10-fold cross-validation for softmax regression with accuracy as a
performance measure, we can write the following piece of code.

```c++
KFoldCV<SoftmaxRegression, Accuracy> cv(10, data, labels, numClasses);
double lambda = 0.1;
double softmaxAccuracy = cv.Evaluate(lambda);
```

Note that the `Evaluate()` method of `KFoldCV` takes any hyperparameters of an
algorithm---that is, anything that is not `data`, `labels`, `numClasses`,
`datasetInfo`, or `weights` (those last three may not be present for every
algorithm type).  To be more specific, in this example the `Evaluate()` method
relies on the following `SoftmaxRegression` constructor:

```c++
template<typename OptimizerType = mlpack::optimization::L_BFGS>
SoftmaxRegression(const arma::mat& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const double lambda = 0.0001,
                  const bool fitIntercept = false,
                  OptimizerType optimizer = OptimizerType());
```

which has the parameter `lambda` after three conventional arguments (`data`,
\c labels and \c numClasses). We can skip passing `fitIntercept` and
`optimizer` since there are the default values.  (Technically, we don't even
need to pass `lambda` since there is a default value.)

In general to cross-validate you need to specify what machine learning algorithm
and metric you are going to use, and then to pass some conventional data-related
parameters into one of the cross-validation constructors and all other
parameters (which are generally hyperparameters) into the `Evaluate()` method.

### 10-fold cross-validation on weighted decision trees

In the following example we will cross-validate `DecisionTree` with weights.
This is very similar to the previous example, except that we also have instance
weights for each point in the dataset.  We can generate weights for the dataset
from the previous example with the code below:

```c++
// Random weights for every point from the code snippet above.
arma::rowvec weights = arma::randu<arma::mat>(1, 100);
```

Given those weights for each point, we can now perform cross-validation by also
passing the weights to the constructor of `KFoldCV`:

```c++
KFoldCV<DecisionTree<>, Accuracy> cv2(10, data, labels, numClasses, weights);
size_t minimumLeafSize = 8;
double weightedDecisionTreeAccuracy = cv2.Evaluate(minimumLeafSize);
```

As with the previous example, internally this call to `cv2.Evaluate()` relies
on the following `DecisionTree` constructor:

```c++
template<typename MatType, typename LabelsType, typename WeightsType>
DecisionTree(MatType&& data,
             LabelsType&& labels,
             const size_t numClasses,
             WeightsType&& weights,
             const size_t minimumLeafSize = 10,
             const std::enable_if_t<arma::is_arma_type<
                 typename std::remove_reference<WeightsType>::type>::value>*
                  = 0);
```

### 10-fold cross-validation with categorical decision trees

`DecisionTree` models can be constructed in multiple other ways. For example, if
we have a dataset with both categorical and numerical features, we can also
perform cross-validation by using the associated `data::DatasetInfo` object.
Thus, given some `data::DatasetInfo` object called `datasetInfo` (that perhaps
was produced by a call to `data::Load()`), we can perform k-fold
cross-validation in a similar manner to the other examples:

```c++
KFoldCV<DecisionTree<>, Accuracy> cv3(10, data, datasetInfo, labels,
    numClasses);
double decisionTreeWithDIAccuracy = cv3.Evaluate(minimumLeafSize);
```

This particular call to `cv3.Evaluate()` relies on the following `DecisionTree`
constructor:

```c++
template<typename MatType, typename LabelsType>
DecisionTree(MatType&& data,
             const data::DatasetInfo& datasetInfo,
             LabelsType&& labels,
             const size_t numClasses,
             const size_t minimumLeafSize = 10);
```

### Simple cross-validation for linear regression

`SimpleCV` has the same interface as `KFoldCV`, except it takes as one of its
arguments a proportion (from 0 to 1) of data used as a validation set. For
example, to validate `LinearRegression` with 20% of the data used in the
validation set we can write the following code.

```c++
// Random responses for every point from the code snippet in the beginning of
// the tutorial.
arma::rowvec responses = arma::randu<arma::rowvec>(100);

SimpleCV<LinearRegression, MSE> cv4(0.2, data, responses);
double lrLambda = 0.05;
double lrMSE = cv4.Evaluate(lrLambda);
```

## Performance measures

The cross-validation classes require a performance measure to be specified.
mlpack has a number of performance measures implemented; below is a list:

 - `Accuracy`: a simple measure of accuracy
 - `F1`: the F1 score; depends on an averaging strategy
 - `MSE`: minimum squared error (for regression problems)
 - `Precision`: the precision, for classification problems
 - `Recall`: the recall, for classification problems

In addition, it is not difficult to implement a custom performance measure.  A
class following the structure below can be used:

```c++
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
```

Once this is implemented, then `CustomMeasure` (or whatever the class is
called) is easy to use as a custom performance measure with `KFoldCV` or
`SimpleCV`.

## The KFoldCV and SimpleCV classes

This section provides details about the `KFoldCV` and `SimpleCV` classes.  The
cross-validation infrastructure is based on heavy amounts of template
metaprogramming, so that any mlpack learner and any performance measure can be
used.  Both classes have two required template parameters and one optional
parameter:

 - `MLAlgorithm`: the type of learner to be used
 - `Metric`: the performance measure to be evaluated
 - `MatType`: the type of matrix used to store the data

In addition, there are two more template parameters, but these are automatically
extracted from the given `MLAlgorithm` class, and users should not need to
specify these parameters except when using an unconventional type like
`arma::fmat` for data points.

The general structure of the `KFoldCV` and `SimpleCV` classes is split into two
parts:

 - The constructor: create the object, and store the data for the `MLAlgorithm`
        training.
 - The `Evaluate()` method: take any non-data parameters for the `MLAlgorithm`
        and calculate the desired performance measure.

This split is important because it defines the API: all data-related parameters
are passed to the constructor, whereas algorithm hyperparameters are passed to
the `Evaluate()` method.

### The KFoldCV and SimpleCV constructors

There are six constructors available for `KFoldCV` and `SimpleCV`, each tailored
for a different learning situation.  Each is given below for the `KFoldCV`
class, but the same constructors are also available for the `SimpleCV` class,
with the exception that instead of specifying `k`, the number of folds, the
`SimpleCV` class takes a parameter between 0 and 1 specifying the percentage of
the dataset to use as a validation set.

 - `KFoldCV(k, xs, ys)`: this is for unweighted regression applications and
        two-class classification applications; `xs` is the dataset and `ys`
        are the responses or labels for each point in the dataset.

 - `KFoldCV(k, xs, ys, numClasses)`: this is for unweighted classification
        applications; `xs` is the dataset, `ys` are the class labels for each
        data point, and `numClasses` is the number of classes in the dataset.

 - `KFoldCV(k, xs, datasetInfo, ys, numClasses)`: this is for unweighted
        categorical/numeric classification applications; `xs` is the dataset,
        `datasetInfo` is a `data::DatasetInfo` object that holds the types of
        each dimension in the dataset, `ys` are the class labels for each data
        point, and `numClasses` is the number of classes in the dataset.

 - `KFoldCV(k, xs, ys, weights)`: this is for weighted regression or
        two-class classification applications; `xs` is the dataset, `ys` are
        the responses or labels for each point in the dataset, and `weights`
        are the weights for each point in the dataset.

 - `KFoldCV(k, xs, ys, numClasses, weights)`: this is for weighted
        classification applications; `xs` is the dataset, `ys` are the class
        labels for each point in the dataset; `numClasses` is the number of
        classes in the dataset, and `weights` holds the weights for each point
        in the dataset.

 - `KFoldCV(k, xs, datasetInfo, ys, numClasses, weights)`: this is for
        weighted cateogrical/numeric classification applications; `xs` is the
        dataset, `datasetInfo` is a `data::DatasetInfo` object that holds the
        types of each dimension in the dataset, `ys` are the class labels for
        each data point, `numClasses` is the number of classes in each dataset,
        and `weights` holds the weights for each point in the dataset.

Note that the constructor you should use is the constructor that most closely
matches the constructor of the machine learning algorithm you would like
performance measures of.  So, for instance, if you are doing multi-class softmax
regression, you could call the constructor `SoftmaxRegression(xs, ys,
numClasses)`.  Therefore, for `KFoldCV` you would call the constructor
`KFoldCV(k, xs, ys, numClasses)` and for `SimpleCV` you would call the
constructor `SimpleCV(pct, xs, ys, numClasses)`.

### The `Evaluate()` method

The other method that `KFoldCV` and `SimpleCV` have is the method to actually
calculate the performance measure: `Evaluate()`.  The `Evaluate()` method takes
any hyperparameters that would follow the data arguments to the constructor or
`Train()` method of the given `MLAlgorithm`.  The `Evaluate()` method takes no
more arguments than that, and returns the desired performance measure on the
dataset.

Therefore, let us suppose that we are interested in cross-validating the
performance of a softmax regression model, and that we have constructed the
appropriate `KFoldCV` object using the code below:

```c++
KFoldCV<SoftmaxRegression, Precision> cv(k, data, labels, numClasses);
```

The `SoftmaxRegression` class has the constructor

```c++
template<typename OptimizerType = mlpack::optimization::L_BFGS>
SoftmaxRegression(const arma::mat& data,
                  const arma::Row<size_t>& labels,
                  const size_t numClasses,
                  const double lambda = 0.0001,
                  const bool fitIntercept = false,
                  OptimizerType optimizer = OptimizerType());
```

Note that all parameters after are `numClasses` are optional.  This means that
we can specify none or any of them in our call to `Evaluate()`.  Below is some
example code showing three different ways we can call `Evaluate()` with the `cv`
object from the code snippet above.

```c++
// First, call with all defaults.
double result1 = cv.Evaluate();

// Next, call with lambda set to 0.1 and fitIntercept set to true.
double result2 = cv.Evaluate(0.1, true);

// Lastly, create a custom optimizer to use for optimization, and use a lambda
// value of 0.5 and fit no intercept.
optimization::SGD<> sgd(0.05, 50000); // Step size of 0.05, 50k max iterations.
double result3 = cv.Evaluate(0.5, false, sgd);
```

The same general idea applies to any `MLAlgorithm`: all hyperparameters must be
passed to the `Evaluate()` method of `KFoldCV` or `SimpleCV`.

## Further references

For further documentation, please see the source code for each of the relevant
classes:

 - `SimpleCV`
 - `KFoldCV`
 - `Accuracy`
 - `F1`
 - `MSE`
 - `Precision`
 - `Recall`

This code is located in `mlpack/core/cv/`.  If you are interested in
implementing a different cross-validation strategy than k-fold cross-validation
or simple cross-validation, take a look at the implementations of each of those
classes to guide your implementation.

In addition, the [hyperparameter tuner](hpt.md) documentation may also be
relevant.
