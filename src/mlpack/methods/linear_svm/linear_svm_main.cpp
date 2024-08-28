/**
 * @file methods/linear_svm/linear_svm_main.cpp
 * @author Yashwant Singh Parihar
 *
 * Main executable for linear svm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME linear_svm

#include <mlpack/core/util/mlpack_main.hpp>

#include "linear_svm.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Linear SVM is an L2-regularized support vector machine.");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of linear SVM for multiclass classification. "
    "Given labeled data, a model can be trained and saved for "
    "future use; or, a pre-trained model can be used to classify new points.");

// Long description.
BINDING_LONG_DESC(
    "An implementation of linear SVMs that uses either L-BFGS or parallel SGD"
    " (stochastic gradient descent) to train the model."
    "\n\n"
    "This program allows loading a linear SVM model (via the " +
    PRINT_PARAM_STRING("input_model") + " parameter) "
    "or training a linear SVM model given training data (specified "
    "with the " + PRINT_PARAM_STRING("training") + " parameter), or both "
    "those things at once.  In addition, this program allows classification on "
    "a test dataset (specified with the " + PRINT_PARAM_STRING("test") + " "
    "parameter) and the classification results may be saved with the " +
    PRINT_PARAM_STRING("predictions") + " output parameter."
    " The trained linear SVM model may be saved using the " +
    PRINT_PARAM_STRING("output_model") + " output parameter."
    "\n\n"
    "The training data, if specified, may have class labels as its last "
    "dimension.  Alternately, the " + PRINT_PARAM_STRING("labels") + " "
    "parameter may be used to specify a separate vector of labels."
    "\n\n"
    "When a model is being trained, there are many options.  L2 regularization "
    "(to prevent overfitting) can be specified with the " +
    PRINT_PARAM_STRING("lambda") + " option, and the number of classes can be "
    "manually specified with the " + PRINT_PARAM_STRING("num_classes") +
    "and if an intercept term is not desired in the model, the " +
    PRINT_PARAM_STRING("no_intercept") + " parameter can be specified."
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
    "optimizers, but the C++ interface must be used to access these."
    "\n\n"
    "Optionally, the model can be used to predict the labels for another "
    "matrix of data points, if " + PRINT_PARAM_STRING("test") + " is "
    "specified.  The " + PRINT_PARAM_STRING("test") + " parameter can be "
    "specified without the " + PRINT_PARAM_STRING("training") + " parameter, "
    "so long as an existing linear SVM model is given with the " +
    PRINT_PARAM_STRING("input_model") + " parameter.  The output predictions "
    "from the linear SVM model may be saved with the " +
    PRINT_PARAM_STRING("predictions") + " parameter.");

// Example.
BINDING_EXAMPLE(
    "As an example, to train a LinaerSVM on the data '" +
    PRINT_DATASET("data") + "' with labels '" + PRINT_DATASET("labels") + "' "
    "with L2 regularization of 0.1, saving the model to '" +
    PRINT_MODEL("lsvm_model") + "', the following command may be used:"
    "\n\n" +
    PRINT_CALL("linear_svm", "training", "data", "labels", "labels",
        "lambda", 0.1, "delta", 1.0, "num_classes", 0,
        "output_model", "lsvm_model") +
    "\n\n"
    "Then, to use that model to predict classes for the dataset '" +
    PRINT_DATASET("test") + "', storing the output predictions in '" +
    PRINT_DATASET("predictions") + "', the following command may be used: "
    "\n\n" +
    PRINT_CALL("linear_svm", "input_model", "lsvm_model", "test", "test",
        "predictions", "predictions"));

// See also...
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("@logistic_regression", "#logistic_regression");
BINDING_SEE_ALSO("LinearSVM on Wikipedia",
    "https://en.wikipedia.org/wiki/Support-vector_machine");
BINDING_SEE_ALSO("LinearSVM C++ class documentation",
    "@doc/user/methods/linear_svm.md");

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set (the matrix "
    "of predictors, X).", "t");
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

class LinearSVMModel
{
 public:
  arma::Col<size_t> mappings;
  LinearSVM<> svm;

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(mappings));
    ar(CEREAL_NVP(svm));
  }
};


// Model loading/saving.
PARAM_MODEL_IN(LinearSVMModel, "input_model", "Existing model "
    "(parameters).", "m");
PARAM_MODEL_OUT(LinearSVMModel, "output_model", "Output for trained "
    "linear svm model.", "M");

// Testing.
PARAM_MATRIX_IN("test", "Matrix containing test dataset.", "T");
PARAM_UROW_IN("test_labels", "Matrix containing test labels.", "L");
PARAM_UROW_OUT("predictions", "If test data is specified, this matrix is where "
    "the predictions for the test set will be saved.", "P");
PARAM_MATRIX_OUT("probabilities", "If test data is specified, this "
    "matrix is where the class probabilities for the test set will be saved.",
    "p");

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

  // One of training and input_model must be specified.
  RequireAtLeastOnePassed(params, { "training", "input_model" }, true);

  // If no output file is given, the user should know that the model will not be
  // saved, but only if a model is being trained.
  RequireAtLeastOnePassed(params, { "output_model", "predictions",
      "probabilities" }, false, "no output will be saved");

  ReportIgnoredParam(params, {{ "test", false }}, "predictions");
  ReportIgnoredParam(params, {{ "test", false }}, "probabilities");
  ReportIgnoredParam(params, {{ "test", false }}, "test_labels");

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

  // Delta must be positive.
  RequireParamValue<int>(params, "epochs", [](int x) { return x > 0; }, true,
      "epochs must be non-negative");

  // These are the matrices we might use.
  arma::mat trainingSet;
  arma::Row<size_t> labels;
  arma::Row<size_t> rawLabels;
  arma::mat testSet;
  arma::Row<size_t> predictedLabels;
  size_t numClasses;

  // Load data matrix.
  if (params.Has("training"))
    trainingSet = std::move(params.Get<arma::mat>("training"));

  // Check if the labels are in a separate file.
  if (params.Has("training") && params.Has("labels"))
  {
    rawLabels = std::move(params.Get<arma::Row<size_t>>("labels"));
    if (trainingSet.n_cols != rawLabels.n_cols)
    {
      Log::Fatal << "The labels must have the same number of points as the "
          << "training dataset." << endl;
    }
  }
  else if (params.Has("training"))
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

  // Load the model, if necessary.
  LinearSVMModel* model;
  if (params.Has("input_model"))
  {
    model = params.Get<LinearSVMModel*>("input_model");
  }
  else
  {
    model = new LinearSVMModel();
  }

  // Now, do the training.
  if (params.Has("training"))
  {
    data::NormalizeLabels(rawLabels, labels, model->mappings);
    numClasses = params.Get<int>("num_classes") == 0 ?
        model->mappings.n_elem : params.Get<int>("num_classes");
    model->svm.Lambda() = lambda;
    model->svm.Delta() = delta;
    model->svm.NumClasses() = numClasses;
    model->svm.FitIntercept() = intercept;

    if (numClasses <= 1)
    {
      if (!params.Has("input_model"))
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
  }
  if (params.Has("test"))
  {
    // Cache the value of GetPrintableParam for the test matrix before we
    // std::move() it.
    std::ostringstream oss;
    oss << params.GetPrintable<arma::mat>("test");
    std::string testOutput = oss.str();

    // Get the test dataset, and get predictions.
    testSet = std::move(params.Get<arma::mat>("test"));
    arma::Row<size_t> predictions;
    size_t trainingDimensionality;

    // Set the dimensionality according to fitintercept.
    if (intercept)
      trainingDimensionality = model->svm.Parameters().n_rows - 1;
    else
      trainingDimensionality = model->svm.Parameters().n_rows;

    // Checking the dimensionality of the test data.
    if (testSet.n_rows != trainingDimensionality)
    {
      // Clean memory if needed.
      if (!params.Has("input_model"))
        delete model;
      Log::Fatal << "Test data dimensionality (" << testSet.n_rows << ") must "
          << "be the same as the dimensionality of the training data ("
          << trainingDimensionality << ")!" << endl;
    }

    // Save class probabilities, if desired.
    if (params.Has("probabilities"))
    {
      Log::Info << "Calculating class probabilities of points in " << testOutput
          << "." << endl;
      arma::mat probabilities;
      model->svm.Classify(testSet, predictions, probabilities);
      params.Get<arma::mat>("probabilities") = std::move(probabilities);
    }

    model->svm.Classify(testSet, predictedLabels);
    data::RevertLabels(predictedLabels, model->mappings, predictions);

    // Calculate accuracy, if desired.
    if (params.Has("test_labels"))
    {
      arma::Row<size_t> testLabels;
      arma::Row<size_t> testRawLabels =
          std::move(params.Get<arma::Row<size_t>>("test_labels"));

      data::NormalizeLabels(testRawLabels, testLabels, model->mappings);

      if (testSet.n_cols != testLabels.n_elem)
      {
        if (!params.Has("input_model"))
          delete model;
        Log::Fatal << "Test data given with " << PRINT_PARAM_STRING("test")
            << " has " << testSet.n_cols << " points, but labels in "
            << PRINT_PARAM_STRING("test_labels") << " have "
            << testLabels.n_elem << " labels!" << endl;
      }

      numClasses = params.Get<int>("num_classes") == 0 ?
          model->mappings.n_elem : params.Get<int>("num_classes");
      arma::Col<size_t> correctClassCounts;
      arma::Col<size_t> labelSize;
      correctClassCounts.zeros(numClasses);
      labelSize.zeros(numClasses);

      for (arma::uword i = 0; i != predictions.n_elem; ++i)
      {
        if (predictions(i) == testLabels(i))
        {
          ++correctClassCounts[testLabels(i)];
        }
        ++labelSize[testLabels(i)];
      }

      size_t totalCorrectClass = 0;
      for (size_t i = 0; i != correctClassCounts.size(); ++i)
      {
        Log::Info << "Accuracy for points with label " << i << " is "
            << (correctClassCounts[i] / static_cast<double>(labelSize[i]))
            << " (" << correctClassCounts[i] << " of " << labelSize[i] << ")."
            << endl;
        totalCorrectClass += correctClassCounts[i];
      }

      Log::Info << "Total accuracy for all points is "
          << (totalCorrectClass) / static_cast<double>(predictions.n_elem)
          << " (" << totalCorrectClass << " of " << predictions.n_elem << ")."
          << endl;
    }

    // Save predictions, if desired.
    if (params.Has("predictions"))
    {
      Log::Info << "Predicting classes of points in '" << testOutput << "'."
          << endl;
      params.Get<arma::Row<size_t>>("predictions") = std::move(predictions);
    }
  }

  params.Get<LinearSVMModel*>("output_model") = model;
}
