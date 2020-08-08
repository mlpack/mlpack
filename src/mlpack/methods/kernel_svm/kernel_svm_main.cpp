/**
 * @file methods/kernel_svm/kernel_svm_main.cpp
 * @author Himanshu Pathak
 *
 * Executable for Kernel SVM.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/kernels/kernel_traits.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/polynomial_kernel.hpp>
#include <mlpack/core/kernels/cosine_distance.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/core/kernels/epanechnikov_kernel.hpp>
#include <mlpack/core/kernels/hyperbolic_tangent_kernel.hpp>
#include <mlpack/core/kernels/laplacian_kernel.hpp>
#include <mlpack/core/kernels/pspectrum_string_kernel.hpp>
#include <mlpack/core/kernels/spherical_kernel.hpp>
#include <mlpack/core/kernels/triangular_kernel.hpp>

#include "kernel_svm.hpp"

using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::kernel;

using namespace std;
using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::util;

PROGRAM_INFO("Kernel SVM is an smo algorithm.",
    // Short description.
    "An implementation of kernel SVM for multiclass classification. "
    "Given labeled data, a model can be trained and saved for "
    "future use; or, a pre-trained model can be used to classify new points.",
    // Long description.
    "An implementation of kerne SVMs that uses many kernels "
    " like polynomial_kernel, linear_kernel and gaussian_kernel etc."
    "\n\n"
    "The kernels that are supported are listed below:"
    "\n\n"
    " * 'linear': the standard linear dot product (same as normal PCA):\n"
    "    K(x, y) = x^T y\n"
    "\n"
    " * 'gaussian': a Gaussian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))\n"
    "\n"
    " * 'polynomial': polynomial kernel; requires offset and degree:\n"
    "    K(x, y) = (x^T y + offset) ^ degree\n"
    "\n"
    " * 'hyptan': hyperbolic tangent kernel; requires scale and offset:\n"
    "    K(x, y) = tanh(scale * (x^T y) + offset)\n"
    "\n"
    " * 'laplacian': Laplacian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y ||) / bandwidth)\n"
    "\n"
    " * 'epanechnikov': Epanechnikov kernel; requires bandwidth:\n"
    "    K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)\n"
    "\n"
    " * 'cosine': cosine distance:\n"
    "    K(x, y) = 1 - (x^T y) / (|| x || * || y ||)\n"
    "\n"
    "The parameters for each of the kernels should be specified with the "
    "options " + PRINT_PARAM_STRING("bandwidth") + ", " +
    PRINT_PARAM_STRING("kernel_scale") + ", " +
    PRINT_PARAM_STRING("offset") + ", or " + PRINT_PARAM_STRING("degree") +
    " (or a combination of those parameters)."
    "\n\n"
    "This program allows loading a kernel SVM model (via the " +
    PRINT_PARAM_STRING("input_model") + " parameter) "
    "or training a kernel SVM model given training data (specified "
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
    "When a model is being trained, there are many options.Sequential minimal "
    "optimization (to prevent overfitting) can be specified with the " +
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
    PRINT_PARAM_STRING("predictions") + " parameter." +
    "\n\n"
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
        "predictions", "predictions"),
    SEE_ALSO("@random_forest", "#random_forest"),
    SEE_ALSO("@logistic_regression", "#logistic_regression"),
    SEE_ALSO("LinearSVM on Wikipedia",
        "https://en.wikipedia.org/wiki/Support-vector_machine"),
    SEE_ALSO("mlpack::svm::LinearSVM C++ class documentation",
        "@doxygen/classmlpack_1_1svm_1_1LinearSVM.html"));

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

template<typename KernelType>
class KernelSVMModel
{
 public:
  arma::Col<size_t> mappings;
  KernelSVM<arma::mat, KernelType> svm;

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(mappings);
    ar & BOOST_SERIALIZATION_NVP(svm);
  }
};

// Model loading/saving.
PARAM_MODEL_IN(KernelSVMModel, "input_model", "Existing model "
    "(parameters).", "m");
PARAM_MODEL_OUT(KernelSVMModel, "output_model", "Output for trained "
    "kernel svm model.", "M");

// Testing.
PARAM_MATRIX_IN("test", "Matrix containing test dataset.", "T");
PARAM_UROW_IN("test_labels", "Matrix containing test labels.", "L");
PARAM_UROW_OUT("predictions", "If test data is specified, this matrix is where "
    "the predictions for the test set will be saved.", "P");
PARAM_MATRIX_OUT("probabilities", "If test data is specified, this "
    "matrix is where the class probabilities for the test set will be saved.",
    "p");

static void mlpackMain()
{
  if (IO::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) IO::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  // Collect command-line options.
  const double lambda = IO::GetParam<double>("lambda");
  const double delta = IO::GetParam<double>("delta");
  const string optimizerType = IO::GetParam<string>("optimizer");
  const double tolerance = IO::GetParam<double>("tolerance");
  const bool intercept = !IO::HasParam("no_intercept");
  const size_t epochs = (size_t) IO::GetParam<int>("epochs");
  const size_t maxIterations = (size_t) IO::GetParam<int>("max_iterations");

  // One of training and input_model must be specified.
  RequireAtLeastOnePassed({ "training", "input_model" }, true);

  // If no output file is given, the user should know that the model will not be
  // saved, but only if a model is being trained.
  RequireAtLeastOnePassed({ "output_model", "predictions", "probabilities"},
      false, "no output will be saved");

  ReportIgnoredParam({{ "test", false }}, "predictions");
  ReportIgnoredParam({{ "test", false }}, "probabilities");
  ReportIgnoredParam({{ "test", false }}, "test_labels");

  // Max Iterations needs to be positive.
  RequireParamValue<int>("max_iterations", [](int x) { return x >= 0; },
      true, "max_iterations must be non-negative");

  // Tolerance needs to be positive.
  RequireParamValue<double>("tolerance", [](double x) { return x >= 0.0; },
      true, "tolerance must be non-negative");

  // Epochs needs to be non-negative.
  RequireParamValue<int>("epochs", [](int x) { return x >= 0; }, true,
      "maximum number of epochs must be non-negative");

  // Get the kernel type and make sure it is valid.
  RequireParamInSet<string>("kernel", { "linear", "gaussian", "polynomial",
      "hyptan", "laplacian", "epanechnikov", "cosine" }, true,
      "unknown kernel type");


  // Load the model, if necessary.
  if (kernelType == "linear")
  {
    KernelSVMModel<LinearKernel>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<KernelSVMModel<LinearKernel>*>("input_model");
    }
    else
    {
      model = new KernelSVMModel<LinearKernel>();
    }
  }
  else if (kernelType == "gaussian")
  {
    const double bandwidth = IO::GetParam<double>("bandwidth");

    KernelSVMModel<GaussianKernel(bandwidth)>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<GaussianKernel(bandwidth)>*("input_model");
    }
    else
    {
      model = new KernelSVMModel<GaussianKernel(bandwidth)>();
    }
  }
  else if (kernelType == "polynomial")
  {
    const double degree = IO::GetParam<double>("degree");
    const double offset = IO::GetParam<double>("offset");

    KernelSVMModel<PolynomialKernel(degree, offset)>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<KernelSVMModel<PolynomialKernel(degree,
                                                offset)>*>("input_model");
    }
    else
    {
      model = new KernelSVMModel<PolynomialKernel(degree, offset)>();
    }
  }
  else if (kernelType == "hyptan")
  {
    const double scale = IO::GetParam<double>("kernel_scale");
    const double offset = IO::GetParam<double>("offset");

    KernelSVMModel<HyperbolicTangentKernel(scale, offset)>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<KernelSVMModel<HyperbolicTangentKernel(scale,
                                               offset)>*>("input_model");
    }
    else
    {
      model = new KernelSVMModel<HyperbolicTangentKernel(scale, offset)>();
    }
  }
  else if (kernelType == "laplacian")
  {
    const double bandwidth = IO::GetParam<double>("bandwidth");

    KernelSVMModel<LaplacianKernel(bandwidth)>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<KernelSVMModel<LaplacianKernel(
      	                                        bandwidth)>*>("input_model");
    }
    else
    {
      model = new KernelSVMModel<LaplacianKernel(bandwidth)>();
    }
  }
  else if (kernelType == "epanechnikov")
  {
    const double bandwidth = IO::GetParam<double>("bandwidth");

    KernelSVMModel<EpanechnikovKernel(bandwidth)>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<KernelSVMModel<EpanechnikovKernel(
      	                                        bandwidth)>*>("input_model");
    }
    else
    {
      model = new KernelSVMModel();
    }
  }
  else if (kernelType == "cosine")
  {
    KernelSVMModel<CosineDistance>* model;
    if (IO::HasParam("input_model"))
    {
      model = IO::GetParam<KernelSVMModel<CosineDistance>*>("input_model");
    }
    else
    {
      model = new KernelSVMModel<CosineDistance>();
    }
  }

  // Now, do the training.
  if (IO::HasParam("training"))
  {
    data::NormalizeLabels(rawLabels, labels, model->mappings);
    numClasses = IO::GetParam<int>("num_classes") == 0 ?
        model->mappings.n_elem : IO::GetParam<int>("num_classes");
    model->svm.Delta() = delta;
    model->svm.NumClasses() = numClasses;
    model->svm.FitIntercept() = intercept;

    if (numClasses <= 1)
    {
      if (!IO::HasParam("input_model"))
        delete model;
      throw std::invalid_argument("Given input data has only 1 class!");
    }

    Log::Info << "Training model with smo." << endl;

    // This will train the model.
    model->svm.Train(trainingSet, labels, maxIterations, tolerance);
  }

  if (IO::HasParam("test"))
  {
    // Cache the value of GetPrintableParam for the test matrix before we
    // std::move() it.
    std::ostringstream oss;
    oss << IO::GetPrintableParam<arma::mat>("test");
    std::string testOutput = oss.str();

    if (!IO::HasParam("training"))
    {
      numClasses = model->svm.NumClasses();
    }
    // Get the test dataset, and get predictions.
    testSet = std::move(IO::GetParam<arma::mat>("test"));
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
      if (!IO::HasParam("input_model"))
        delete model;
      Log::Fatal << "Test data dimensionality (" << testSet.n_rows << ") must "
          << "be the same as the dimensionality of the training data ("
          << trainingDimensionality << ")!" << endl;
    }

    // Save class probabilities, if desired.
    if (IO::HasParam("probabilities"))
    {
      Log::Info << "Calculating class probabilities of points in " << testOutput
          << "." << endl;
      arma::mat probabilities;
      model->svm.Classify(testSet, probabilities);
      IO::GetParam<arma::mat>("probabilities") = std::move(probabilities);
    }

    model->svm.Classify(testSet, predictedLabels);
    data::RevertLabels(predictedLabels, model->mappings, predictions);

    // Calculate accuracy, if desired.
    if (IO::HasParam("test_labels"))
    {
      arma::Row<size_t> testLabels;
      arma::Row<size_t> testRawLabels =
          std::move(IO::GetParam<arma::Row<size_t>>("test_labels"));

      data::NormalizeLabels(testRawLabels, testLabels, model->mappings);

      if (testSet.n_cols != testLabels.n_elem)
      {
        if (!IO::HasParam("input_model"))
          delete model;
        Log::Fatal << "Test data given with " << PRINT_PARAM_STRING("test")
            << " has " << testSet.n_cols << " points, but labels in "
            << PRINT_PARAM_STRING("test_labels") << " have "
            << testLabels.n_elem << " labels!" << endl;
      }

      numClasses = IO::GetParam<int>("num_classes") == 0 ?
          model->mappings.n_elem : IO::GetParam<int>("num_classes");
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
    if (IO::HasParam("predictions"))
    {
      Log::Info << "Predicting classes of points in '" << testOutput << "'."
          << endl;
      IO::GetParam<arma::Row<size_t>>("predictions") = std::move(predictions);
    }
  }

  IO::GetParam<KernelSVMModel*>("output_model") = model;
}
