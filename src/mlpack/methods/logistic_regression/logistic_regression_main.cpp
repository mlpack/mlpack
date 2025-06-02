/**
 * @file methods/logistic_regression/logistic_regression_main.cpp
 * @author Ryan Curtin
 *
 * Main executable for logistic regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME logistic_regression

#include <mlpack/core/util/mlpack_main.hpp>

#include "logistic_regression.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("L2-regularized Logistic Regression and Prediction");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of L2-regularized logistic regression for two-class "
    "classification.  Given labeled data, a model can be trained and saved for "
    "future use; or, a pre-trained model can be used to classify new points.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of L2-regularized logistic regression using either the "
    "L-BFGS optimizer or SGD (stochastic gradient descent).  This solves the "
    "regression problem"
    "\n\n"
    "  y = (1 / 1 + e^-(X * b))."
    "\n\n"
    "In this setting, y corresponds to class labels and X corresponds to data."
    "\n\n"
    "This program allows loading a logistic regression model (via the " +
    PRINT_PARAM_STRING("input_model") + " parameter) "
    "or training a logistic regression model given training data (specified "
    "with the " + PRINT_PARAM_STRING("training") + " parameter), or both "
    "those things at once.  In addition, this program allows classification on "
    "a test dataset (specified with the " + PRINT_PARAM_STRING("test") + " "
    "parameter) and the classification results may be saved with the " +
    PRINT_PARAM_STRING("predictions") + " output parameter."
    " The trained logistic regression model may be saved using the " +
    PRINT_PARAM_STRING("output_model") + " output parameter."
    "\n\n"
    "The training data, if specified, may have class labels as its last "
    "dimension.  Alternately, the " + PRINT_PARAM_STRING("labels") + " "
    "parameter may be used to specify a separate matrix of labels."
    "\n\n"
    "When a model is being trained, there are many options.  L2 regularization "
    "(to prevent overfitting) can be specified with the " +
    PRINT_PARAM_STRING("lambda") + " option, and the "
    "optimizer used to train the model can be specified with the " +
    PRINT_PARAM_STRING("optimizer") + " parameter.  Available options are "
    "'sgd' (stochastic gradient descent) and 'lbfgs' (the L-BFGS optimizer).  "
    "There are also various parameters for the optimizer; the " +
    PRINT_PARAM_STRING("max_iterations") + " parameter specifies the maximum "
    "number of allowed iterations, and the " +
    PRINT_PARAM_STRING("tolerance") + " parameter specifies the tolerance for "
    "convergence.  For the SGD optimizer, the " +
    PRINT_PARAM_STRING("step_size") + " parameter controls the step size taken "
    "at each iteration by the optimizer.  The batch size for SGD is controlled "
    "with the " + PRINT_PARAM_STRING("batch_size") + " parameter. If the "
    "objective function for your data is oscillating between Inf and 0, the "
    "step size is probably too large.  There are more parameters for the "
    "optimizers, but the C++ interface must be used to access these."
    "\n\n"
    "For SGD, an iteration refers to a single point. So to take a single pass "
    "over the dataset with SGD, " + PRINT_PARAM_STRING("max_iterations") +
    " should be set to the number of points in the dataset."
    "\n\n"
    "Optionally, the model can be used to predict the responses for another "
    "matrix of data points, if " + PRINT_PARAM_STRING("test") + " is "
    "specified.  The " + PRINT_PARAM_STRING("test") + " parameter can be "
    "specified without the " + PRINT_PARAM_STRING("training") + " parameter, "
    "so long as an existing logistic regression model is given with the " +
    PRINT_PARAM_STRING("input_model") + " parameter.  The output predictions "
    "from the logistic regression model may be saved with the " +
    PRINT_PARAM_STRING("predictions") + " parameter." +
    "\n\n"
    "This implementation of logistic regression does not support the general "
    "multi-class case but instead only the two-class case.  Any labels must be "
    "either " + STRINGIFY(BINDING_MIN_LABEL) + " or " +
    std::to_string(BINDING_MIN_LABEL + 1) + ".  For more classes, see the "
    "softmax regression implementation.");

// Example.
BINDING_EXAMPLE(
    "As an example, to train a logistic regression model on the data '" +
    PRINT_DATASET("data") + "' with labels '" + PRINT_DATASET("labels") + "' "
    "with L2 regularization of 0.1, saving the model to '" +
    PRINT_MODEL("lr_model") + "', the following command may be used:"
    "\n\n" +
    PRINT_CALL("logistic_regression", "training", "data", "labels", "labels",
        "lambda", 0.1, "output_model", "lr_model", "print_training_accuracy",
        true) +
    "\n\n"
    "Then, to use that model to predict classes for the dataset '" +
    PRINT_DATASET("test") + "', storing the output predictions in '" +
    PRINT_DATASET("predictions") + "', the following command may be used: "
    "\n\n" +
    PRINT_CALL("logistic_regression", "input_model", "lr_model", "test", "test",
        "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Logistic regression on Wikipedia",
    "https://en.wikipedia.org/wiki/Logistic_regression");
BINDING_SEE_ALSO(":LogisticRegression C++ class documentation",
    "@doc/user/methods/logistic_regression.md");

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set (the matrix "
    "of predictors, X).", "t");
PARAM_UROW_IN("labels", "A matrix containing labels (0 or 1) for the points "
    "in the training set (y).", "l");

// Optimizer parameters.
PARAM_DOUBLE_IN("lambda", "L2-regularization parameter for training.", "L",
    0.0);
PARAM_STRING_IN("optimizer", "Optimizer to use for training ('lbfgs' or "
    "'sgd').", "O", "lbfgs");
PARAM_DOUBLE_IN("tolerance", "Convergence tolerance for optimizer.", "e",
    1e-10);
PARAM_INT_IN("max_iterations", "Maximum iterations for optimizer (0 indicates "
    "no limit).", "n", 10000);
PARAM_DOUBLE_IN("step_size", "Step size for SGD optimizer.",
    "s", 0.01);
PARAM_INT_IN("batch_size", "Batch size for SGD.", "b", 64);

// Model loading/saving.
PARAM_MODEL_IN(LogisticRegression<>, "input_model", "Existing model "
    "(parameters).", "m");
PARAM_MODEL_OUT(LogisticRegression<>, "output_model", "Output for trained "
    "logistic regression model.", "M");

// Testing.
PARAM_MATRIX_IN("test", "Matrix containing test dataset.", "T");
PARAM_UROW_OUT("predictions", "If test data is specified, this matrix is where "
    "the predictions for the test set will be saved.", "P");
PARAM_MATRIX_OUT("probabilities", "If test data is specified, this "
    "matrix is where the class probabilities for the test set will be saved.",
    "p");
PARAM_DOUBLE_IN("decision_boundary", "Decision boundary for prediction; if the "
    "logistic function for a point is less than the boundary, the class is "
    "taken to be 0; otherwise, the class is 1.", "d", 0.5);
PARAM_FLAG("print_training_accuracy", "If set, then the accuracy of the model "
    "on the training set will be printed (verbose must also be specified).",
    "a");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Collect command-line options.
  const double lambda = params.Get<double>("lambda");
  const string optimizerType = params.Get<string>("optimizer");
  const double tolerance = params.Get<double>("tolerance");
  const double stepSize = params.Get<double>("step_size");
  const size_t batchSize = (size_t) params.Get<int>("batch_size");
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");
  const double decisionBoundary = params.Get<double>("decision_boundary");

  // One of training and input_model must be specified.
  RequireAtLeastOnePassed(params, { "training", "input_model" }, true);

  // If no output file is given, the user should know that the model will not be
  // saved, but only if a model is being trained.
  if (params.Has("training"))
  {
    RequireAtLeastOnePassed(params, { "output_model" }, false, "trained model "
        "will not be saved");
  }

  RequireAtLeastOnePassed(params, { "output_model", "predictions",
      "probabilities"}, false, "no output will be saved");

  ReportIgnoredParam(params, {{ "test", false }}, "predictions");
  ReportIgnoredParam(params, {{ "test", false }}, "probabilities");

  ReportIgnoredParam(params, {{ "training", false }},
      "print_training_accuracy");

  RequireAtLeastOnePassed(params,
      { "test", "output_model", "print_training_accuracy" }, false,
      "the trained logistic regression model will not be used or saved");

  // Max Iterations needs to be positive.
  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "max_iterations must be positive or zero");

  // Batch Size needs to be greater than zero.
  RequireParamValue<int>(params, "batch_size", [](int x) { return x > 0; },
      true, "batch_size must be greater than zero");

  // Tolerance needs to be positive.
  RequireParamValue<double>(params, "tolerance",
      [](double x) { return x >= 0.0; },
      true, "tolerance must be positive or zero");

  // Optimizer has to be L-BFGS or SGD.
  RequireParamInSet<string>(params, "optimizer", { "lbfgs", "sgd" },
      true, "unknown optimizer");

  // Lambda must be positive.
  RequireParamValue<double>(params, "lambda", [](double x) { return x >= 0.0; },
      true, "lambda must be positive or zero");

  // Decision boundary must be between 0 and 1.
  RequireParamValue<double>(params, "decision_boundary",
      [](double x) { return x >= 0.0 && x <= 1.0; }, true,
      "decision boundary must be between 0.0 and 1.0");

  RequireParamValue<double>(params, "step_size",
      [](double x) { return x >= 0.0; }, true, "step size must be positive");

  if (optimizerType != "sgd")
  {
    if (params.Has("step_size"))
    {
      Log::Warn << PRINT_PARAM_STRING("step_size") << " ignored because "
          << "optimizer type is not 'sgd'." << std::endl;
    }
    if (params.Has("batch_size"))
    {
      Log::Warn << PRINT_PARAM_STRING("batch_size") << " ignored because "
          << "optimizer type is not 'sgd'." << std::endl;
    }
  }

  // These are the matrices we might use.
  arma::mat regressors;
  arma::Row<size_t> responses;
  arma::mat testSet;
  arma::Row<size_t> predictions;

  // Load data matrix.
  if (params.Has("training"))
    regressors = std::move(params.Get<arma::mat>("training"));

  // Load the model, if necessary.
  LogisticRegression<>* model;
  if (params.Has("input_model"))
    model = params.Get<LogisticRegression<>*>("input_model");
  else
  {
    model = new LogisticRegression<>(0, 0);

    // Set the size of the parameters vector, if necessary.
    if (!params.Has("labels"))
      model->Parameters() = zeros<arma::rowvec>(regressors.n_rows);
    else
      model->Parameters() = zeros<arma::rowvec>(regressors.n_rows + 1);
  }

  // Check if the responses are in a separate file.
  if (params.Has("training") && params.Has("labels"))
  {
    responses = std::move(params.Get<arma::Row<size_t>>("labels"));
    if (responses.n_cols != regressors.n_cols)
    {
      // Clean memory if needed.
      if (!params.Has("input_model"))
        delete model;

      Log::Fatal << "The labels must have the same number of points as the "
          << "training dataset." << endl;
    }
  }
  else if (params.Has("training"))
  {
    // Checking the size of training data if no labels are passed.
    if (regressors.n_rows < 2)
    {
      // Clean memory if needed.
      if (!params.Has("input_model"))
        delete model;

      Log::Fatal << "Can't get responses from training data since it has less "
          << "than 2 rows." << endl;
    }

    // The initial predictors for y, Nx1.
    responses = ConvTo<arma::Row<size_t>>::From(
        regressors.row(regressors.n_rows - 1));
    regressors.shed_row(regressors.n_rows - 1);
  }

  // Verify the labels.
  if (params.Has("training") && max(responses) > 1)
  {
    // Clean memory if needed.
    if (!params.Has("input_model"))
      delete model;

    Log::Fatal << "The labels must be either 0 or 1, not " << max(responses)
        << "!" << endl;
  }

  // Now, do the training.
  if (params.Has("training"))
  {
    model->Lambda() = lambda;

    if (optimizerType == "sgd")
    {
      ens::SGD<> sgdOpt;
      sgdOpt.MaxIterations() = maxIterations;
      sgdOpt.Tolerance() = tolerance;
      sgdOpt.StepSize() = stepSize;
      sgdOpt.BatchSize() = batchSize;
      Log::Info << "Training model with SGD optimizer." << endl;

      // This will train the model.
      timers.Start("logistic_regression_optimization");
      model->Train(regressors, responses, sgdOpt);
      timers.Stop("logistic_regression_optimization");
    }
    else if (optimizerType == "lbfgs")
    {
      ens::L_BFGS lbfgsOpt;
      lbfgsOpt.MaxIterations() = maxIterations;
      lbfgsOpt.MinGradientNorm() = tolerance;
      Log::Info << "Training model with L-BFGS optimizer." << endl;

      // This will train the model.
      timers.Start("logistic_regression_optimization");
      model->Train(regressors, responses, lbfgsOpt);
      timers.Stop("logistic_regression_optimization");
    }

    // Did we want training accuracy?
    if (params.Has("print_training_accuracy"))
    {
      timers.Start("lr_prediction");
      arma::Row<size_t> predictions;
      model->Classify(regressors, predictions);

      const size_t correct = accu(predictions == responses);

      Log::Info << correct << " of " << responses.n_elem << " correct on "
          << "training set ("
          << (double(correct) / double(responses.n_elem) * 100) << ")." << endl;
      timers.Stop("lr_prediction");
    }
  }

  if (params.Has("test"))
  {
    const arma::mat& testSet = params.Get<arma::mat>("test");

    // Checking the dimensionality of the test data.
    if (testSet.n_rows != model->Parameters().n_cols - 1)
    {
      // Clean memory if needed.
      const size_t trainingDimensionality = model->Parameters().n_cols - 1;
      if (!params.Has("input_model"))
        delete model;

      Log::Fatal << "Test data dimensionality (" << testSet.n_rows << ") must "
          << "be the same as the dimensionality of the training data ("
          << trainingDimensionality << ")!" << endl;
    }

    // We must perform predictions on the test set.  Training (and the
    // optimizer) are irrelevant here; we'll pass in the model we have.
    if (params.Has("predictions"))
    {
      Log::Info << "Predicting classes of points in '"
          << params.GetPrintable<arma::mat>("test") << "'." << endl;
      model->Classify(testSet, predictions, decisionBoundary);

      if (params.Has("predictions"))
        params.Get<arma::Row<size_t>>("predictions") = predictions;
    }

    if (params.Has("probabilities"))
    {
      Log::Info << "Calculating class probabilities of points in '"
          << params.GetPrintable<arma::mat>("test") << "'." << endl;
      arma::mat probabilities;
      model->Classify(testSet, predictions, probabilities);

      if (params.Has("probabilities"))
        params.Get<arma::mat>("probabilities") = std::move(probabilities);
    }
  }

  params.Get<LogisticRegression<>*>("output_model") = model;
}
