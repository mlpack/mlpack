/**
 * @file methods/softmax_regression/softmax_regression_train_main.cpp
 *
 * Implementation of softmax regression classification training step.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME softmax_regression_train

#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression_utils.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Softmax Regression");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of softmax regression for classification, which is a "
    "multiclass generalization of logistic regression.  Given labeled data, a "
    "softmax regression model can be trained for future use of "
    "classification on new points.");

// Long description.
BINDING_LONG_DESC(
    "Implementation of softmax regression, a generalization of logistic "
    "regression to the multiclass case, with support for L2 regularization. "
    "\n\n"
    "Training a softmax regression model is done by giving a file of training "
    "points with the " + PRINT_PARAM_STRING("training") + " parameter and their"
    " corresponding labels with the " + PRINT_PARAM_STRING("labels") +
    " parameter. The number of classes can be manually specified with the " +
    PRINT_PARAM_STRING("number_of_classes") + " parameter, and the maximum " +
    "number of iterations of the L-BFGS optimizer can be specified with the " +
    PRINT_PARAM_STRING("max_iterations") + " parameter.  The L2 regularization "
    "constant can be specified with the " + PRINT_PARAM_STRING("lambda") +
    " parameter and if an intercept term is not desired in the model, the " +
    PRINT_PARAM_STRING("no_intercept") + " parameter can be specified."
    "\n\n");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("softmax_regression", "train", "classify", "probabilities") +
    "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "softmax_regression") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train",
                "lambda", 0.1));

// See also...
BINDING_SEE_ALSO("@logistic_regression", "#logistic_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Multinomial logistic regression (softmax regression) on "
    "Wikipedia",
    "https://en.wikipedia.org/wiki/Multinomial_logistic_regression");
BINDING_SEE_ALSO("SoftmaxRegression C++ class documentation",
    "@doc/user/methods/softmax_regression.md");

// Required options.
PARAM_MATRIX_IN_REQ("training", "A matrix containing the training set (the "
    "matrix of predictors, X).", "t");
PARAM_UROW_IN_REQ("labels", "A matrix containing labels (0 or 1) for the "
    "points in the training set (y). The labels must order as a row.", "l");

// Model output.
PARAM_MODEL_OUT(SoftmaxRegression<>, "output_model", "File to save trained "
    "softmax regression model to.", "M");

// Softmax configuration options.
PARAM_INT_IN("max_iterations", "Maximum number of iterations before "
    "termination.", "n", 400);

PARAM_INT_IN("number_of_classes", "Number of classes for classification; if "
    "unspecified (or 0), the number of classes found in the labels will be "
    "used.", "c", 0);

PARAM_DOUBLE_IN("lambda", "L2-regularization constant", "r", 0.0001);

PARAM_FLAG("no_intercept", "Do not add the intercept term to the model.", "N");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  const int maxIterations = params.Get<int>("max_iterations");

  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "maximum number of iterations must be greater than or equal to 0");
  RequireParamValue<double>(params, "lambda", [](double x) { return x >= 0.0; },
      true, "lambda penalty parameter must be greater than or equal to 0");
  RequireParamValue<int>(params, "number_of_classes",
      [](int x) { return x >= 0; }, true, "number of classes must be greater "
      "than or equal to 0 (equal to 0 in case of unspecified.)");

  arma::mat trainData = std::move(params.Get<arma::mat>("training"));
  arma::Row<size_t> trainLabels =
    std::move(params.Get<arma::Row<size_t>>("labels"));

  if (trainData.n_cols != trainLabels.n_elem)
    Log::Fatal << "Samples of input_data should same as the size of "
        << "input_label." << endl;

  size_t numClasses = (size_t) params.Get<int>("number_of_classes");
  if (numClasses == 0)
  {
    set<size_t> unique_labels(begin(trainLabels), end(trainLabels));
    numClasses = unique_labels.size();
  }

  const bool intercept = params.Has("no_intercept") ? false : true;

  const size_t numBasis = 5;
  ens::L_BFGS optimizer(numBasis, maxIterations);
  timers.Start("softmax_regression_optimization");
  SoftmaxRegression<>* sm = new SoftmaxRegression<>(trainData, trainLabels,
      numClasses, params.Get<double>("lambda"), intercept,
      std::move(optimizer));
  timers.Stop("softmax_regression_optimization");

  params.Get<SoftmaxRegression<>*>("output_model") = sm;
}
