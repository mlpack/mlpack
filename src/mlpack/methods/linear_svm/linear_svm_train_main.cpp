/**
 * @file methods/linear_svm/linear_svm_main.cpp
 * @author Yashwant Singh Parihar
 * @author Dirk Eddelbuettel
 *
 * Train a Linear SVM model given data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME linear_svm_train

#include <mlpack/core/util/mlpack_main.hpp>
#include "linear_svm.hpp"
#include "linear_svm_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Linear SVM Training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of linear SVM for multiclass classification. "
    "Given labeled data, a model is.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of linear SVMs that uses either L-BFGS or parallel SGD"
    " (stochastic gradient descent) to train the model."
    "\n\n"
    "This implementation allows training a linear SVM model given training "
    "data (specified with the " + PRINT_PARAM_STRING("training") +
    " parameter)."
    "\n\n"
    "The training data may have class labels as its last dimension. "
    "Alternately, the " + PRINT_PARAM_STRING("labels") + " "
    "parameter may be used to specify a separate vector of labels."
    "\n\n"
    "When a model is being trained, there are many options.  L2 regularization "
    "(to prevent overfitting) can be specified with the " +
    PRINT_PARAM_STRING("lambda") + " option, and the number of classes can be "
    "manually specified with the " + PRINT_PARAM_STRING("num_classes") +
    "and if an intercept term is not desired in the model, the " +
    PRINT_PARAM_STRING("no_intercept") + " parameter can be specified."
    "\n\n"
    "Margin of difference between correct class and other classes can "
    "be specified with the " + PRINT_PARAM_STRING("delta") + " option."
    "The optimizer used to train the model can be specified with the " +
    PRINT_PARAM_STRING("optimizer") + " parameter.  Available options are "
    "'psgd' (parallel stochastic gradient descent) and 'lbfgs' (the L-BFGS"
    " optimizer).  There are also various parameters for the optimizer; the " +
    PRINT_PARAM_STRING("max_iterations") + " parameter specifies the maximum "
    "number of allowed iterations, and the " +
    PRINT_PARAM_STRING("tolerance") + " parameter specifies the tolerance for "
    "convergence.  For the parallel SGD optimizer, the " +
    PRINT_PARAM_STRING("step_size") + " parameter controls the step size taken "
    "at each iteration by the optimizer and the maximum number of epochs "
    "(specified with " + PRINT_PARAM_STRING("epochs") + "). If the "
    "objective function for your data is oscillating between Inf and 0, the "
    "step size is probably too large.  There are more parameters for the "
    "optimizers, but the C++ interface must be used to access these.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("linear_svm", "train", "classify", "scores") + "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "linear_svm") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train",
        "lambda", 0.1, "delta", 1.0, "num_classes", 0));

// See also...
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("@logistic_regression", "#logistic_regression");
BINDING_SEE_ALSO("LinearSVM on Wikipedia",
    "https://en.wikipedia.org/wiki/Support-vector_machine");
BINDING_SEE_ALSO("LinearSVM C++ class documentation",
    "@doc/user/methods/linear_svm.md");

// Training parameters.
PARAM_MATRIX_IN_REQ("training", "A matrix containing the training set (the "
    "matrix of predictors, X).", "t");
PARAM_UROW_IN("labels", "A matrix containing labels (0 or 1) for the points "
    "in the training set (y).", "l");

// Optimizer parameters.
PARAM_DOUBLE_IN("lambda", "L2-regularization parameter for training.", "r",
    0.0001);
PARAM_DOUBLE_IN("delta", "Margin of difference between correct class and other "
    "classes.", "d", 1.0);
PARAM_INT_IN("num_classes", "Number of classes for classification; if "
    "unspecified (or 0), the number of classes found in the labels will be "
    "used.", "c", 0);
PARAM_FLAG("no_intercept", "Do not add the intercept term to the model.", "N");
PARAM_STRING_IN("optimizer", "Optimizer to use for training ('lbfgs' or "
    "'psgd').", "O", "lbfgs");
PARAM_DOUBLE_IN("tolerance", "Convergence tolerance for optimizer.", "e",
    1e-10);
PARAM_INT_IN("max_iterations", "Maximum iterations for optimizer (0 indicates "
    "no limit).", "n", 10000);
PARAM_DOUBLE_IN("step_size", "Step size for parallel SGD optimizer.",
    "a", 0.01);
PARAM_FLAG("shuffle", "Don't shuffle the order in which data points are "
    "visited for parallel SGD.", "S");
PARAM_INT_IN("epochs", "Maximum number of full epochs over dataset for "
    "psgd", "E", 50);
PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);

// Model saving.
PARAM_MODEL_OUT(LinearSVMModel, "output_model", "Output for trained "
    "linear svm model.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  if (params.Get<int>("seed") != 0)
    RandomSeed((size_t) params.Get<int>("seed"));
  else
    RandomSeed((size_t) std::time(NULL));

  // Collect command-line options.
  const double lambda = params.Get<double>("lambda");
  const double delta = params.Get<double>("delta");
  const string optimizerType = params.Get<string>("optimizer");
  const double tolerance = params.Get<double>("tolerance");
  const bool intercept = !params.Has("no_intercept");
  const size_t epochs = (size_t) params.Get<int>("epochs");
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");

  // Max Iterations needs to be positive.
  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "max_iterations must be non-negative");

  // Tolerance needs to be positive.
  RequireParamValue<double>(params, "tolerance",
      [](double x) { return x >= 0.0; },
      true, "tolerance must be non-negative");

  // Optimizer has to be L-BFGS or parallel SGD.
  RequireParamInSet<string>(params, "optimizer", { "lbfgs", "psgd" },
      true, "unknown optimizer");

  // Epochs needs to be non-negative.
  RequireParamValue<int>(params, "epochs", [](int x) { return x >= 0; }, true,
      "maximum number of epochs must be non-negative");

  if (optimizerType != "psgd")
  {
    if (params.Has("step_size"))
    {
      Log::Warn << PRINT_PARAM_STRING("step_size") << " ignored because "
          << "optimizer type is not 'psgd'." << std::endl;
    }
    if (params.Has("shuffle"))
    {
      Log::Warn << PRINT_PARAM_STRING("shuffle") << " ignored because "
          << "optimizer type is not 'psgd'." << std::endl;
    }
    if (params.Has("epochs"))
    {
      Log::Warn << PRINT_PARAM_STRING("epochs") << " ignored because "
          << "optimizer type is not 'psgd'." << std::endl;
    }
  }

  if (optimizerType != "lbfgs")
  {
    if (params.Has("max_iterations"))
    {
      Log::Warn << PRINT_PARAM_STRING("max_iterations") << " ignored because "
          << "optimizer type is not 'lbfgs'." << std::endl;
    }
  }

  // Step Size must be positive.
  RequireParamValue<double>(params, "step_size",
      [](double x) { return x > 0.0; }, true, "step size must be positive");

  // Lambda must be positive.
  RequireParamValue<double>(params, "lambda", [](double x) { return x >= 0.0; },
      true, "lambda must be non-negative");

  // Number of Classes must be Non-Negative
  RequireParamValue<int>(params, "num_classes", [](int x) { return x >= 0; },
                         true, "number of classes must be greater than or "
                         "equal to 0 (equal to 0 in case of unspecified.)");

  // Delta must be positive.
  RequireParamValue<double>(params, "delta", [](double x) { return x >= 0.0; },
      true, "delta must be non-negative");

  // Epochs must be positive.
  RequireParamValue<int>(params, "epochs", [](int x) { return x > 0; }, true,
      "epochs must be non-negative");

  // These are the matrices we might use.
  arma::mat trainingSet;
  arma::Row<size_t> labels;
  arma::Row<size_t> rawLabels;
  size_t numClasses;

  // Load data matrix.
  trainingSet = std::move(params.Get<arma::mat>("training"));

  // Check if the labels are in a separate argument.
  if (params.Has("labels"))
  {
    rawLabels = std::move(params.Get<arma::Row<size_t>>("labels"));
    if (trainingSet.n_cols != rawLabels.n_cols)
    {
      Log::Fatal << "The labels must have the same number of points as the "
          << "training dataset." << endl;
    }
  }
  else
  {
    // Checking the size of training data if no labels are passed.
    if (trainingSet.n_rows < 2)
    {
      Log::Fatal << "Can't get labels from training data since it has less "
          << "than 2 rows." << endl;
    }

    // The initial predictors for y, Nx1.
    rawLabels = ConvTo<arma::Row<size_t>>::From(
        trainingSet.row(trainingSet.n_rows - 1));
    trainingSet.shed_row(trainingSet.n_rows - 1);
  }

  LinearSVMModel* model = new LinearSVMModel();

  // Now do the training.

  NormalizeLabels(rawLabels, labels, model->mappings);
  numClasses = params.Get<int>("num_classes") == 0 ?
      model->mappings.n_elem : params.Get<int>("num_classes");
  model->svm.Lambda() = lambda;
  model->svm.Delta() = delta;
  model->svm.NumClasses() = numClasses;
  model->svm.FitIntercept() = intercept;

  if (numClasses <= 1)
  {
    delete model;
    throw std::invalid_argument("Given input data has only 1 class!");
  }

  if (optimizerType == "lbfgs")
  {
    ens::L_BFGS lbfgsOpt;
    lbfgsOpt.MaxIterations() = maxIterations;
    lbfgsOpt.MinGradientNorm() = tolerance;

    Log::Info << "Training model with L-BFGS optimizer." << endl;

    // This will train the model.
    timers.Start("linear_svm_optimization");
    model->svm.Train(trainingSet, labels, numClasses, lbfgsOpt);
    timers.Stop("linear_svm_optimization");
  }
  else if (optimizerType == "psgd")
  {
    const double stepSize = params.Get<double>("step_size");
    const bool shuffle = !params.Has("shuffle");
    const size_t maxIt = epochs * trainingSet.n_cols;

    ens::ConstantStep decayPolicy(stepSize);

    #ifdef MLPACK_USE_OPENMP
    size_t threads = omp_get_max_threads();
    #else
    size_t threads = 1;
    Log::Warn << "Using parallel SGD, but OpenMP support is "
              << "not available!" << endl;
    #endif

    ens::ParallelSGD<ens::ConstantStep> psgdOpt(maxIt, std::ceil(
      (float) trainingSet.n_cols / threads), tolerance, shuffle,
      decayPolicy);

    Log::Info << "Training model with ParallelSGD optimizer." << endl;

    // This will train the model.
    timers.Start("linear_svm_optimization");
    model->svm.Train(trainingSet, labels, numClasses, psgdOpt);
    timers.Stop("linear_svm_optimization");
  }

  params.Get<LinearSVMModel*>("output_model") = model;
}
