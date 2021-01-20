/**
 * @file bayesian_linear_regression_main.cpp
 * @author Clement Mercier
 *
 * Executable for BayesianLinearRegression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "rvm_regression.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;

// Program Name.
BINDING_NAME("Relevance Vector Machine for regression");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Relevance Vector Machine (RVM) that can also be "
    "used for ARD regression on a given dataset if the kernel is not specified."
    );

// Long description.
BINDING_LONG_DESC(
    "This program trains a RVM model for regression on the dataset provided "
    "with the specified kernel. RVM is a bayesian kernel based technique "
    "similar to the SVM whose the solution is much more sparse, making this "
    "model fast to apply on test data. The optimization procedure maximizes "
    "the log marginal likelihood leading to automatic determination of the "
    "the best hyperparameters set associated to the relevant vectors."
    "\n\n"
    "To train a RVMRegression model, the " +
    PRINT_PARAM_STRING("input") + " and " + PRINT_PARAM_STRING("responses") +
    "parameters must be given. The " + PRINT_PARAM_STRING("center") +
    "and " + PRINT_PARAM_STRING("scale") + " parameters control the "
    "centering and the normalizing options. A trained model can be saved with "
    "the " + PRINT_PARAM_STRING("output_model") + ". If no training is desired "
    "at all, a model can be passed via the " +
    PRINT_PARAM_STRING("input_model") + " parameter."
    "\n\n"
    "The program can also provide predictions for test data using either the "
    "trained model or the given input model. Test points can be specified "
    "with the " + PRINT_PARAM_STRING("test") + " parameter.  Predicted "
    "responses to the test points can be saved with the " +
    PRINT_PARAM_STRING("predictions") + " output parameter. The "
    "corresponding standard deviations can be saved by precising the " +
    PRINT_PARAM_STRING("stds") + " parameter."
    "If the " + PRINT_PARAM_STRING("kernel") + "is not specified the model "
    "optimized is a bayesian linear regression associated to an ARD prior "
    "leading sparse solution over the variable domain."
    "\n"
    "The supported kernel are listed below:"
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
    " (or a combination of those parameters).");

//Example
BINDING_EXAMPLE(
    "For example, the following command trains a model on the data " +
    PRINT_DATASET("data") + " and responses " + PRINT_DATASET("responses") +
    "with center and scale set to true and a gaussian kernel of "
    "bandwith 1.0. RVM is solved and the model is saved to " +
    PRINT_MODEL("rvm_regression") + ":"
    "\n\n" +
    PRINT_CALL("rvm_regression", "input", "data", "responses", "responses", 
               "center", 1, "scale", 1, "output_model", 
               "rvm_regression_model", "kernel", "gaussian", "bandwidth", 1.0) +
    "The following command uses the " + PRINT_MODEL("rvm_regression_model") + 
    "to provide predicted responses for the data " + PRINT_DATASET("test") + 
    "and save those responses to " + PRINT_DATASET("test_predictions") + ":"
    "\n\n" + 
    PRINT_CALL("rvm_regression", "input_model", "rvm_regression_model", "test", 
               "test", "predictions", "test_predictions") +
    "\n\n"
    "Because the estimator computes a predictive distribution instead of "
    "simple point estimate, the " + PRINT_PARAM_STRING("stds") + " parameter "
    "allows to save the prediction uncertainties: "
    "\n\n" +
    PRINT_CALL("rvm_regression", "input_model",
               "rvm_regression_model", "test", "test",
               "predictions", "test_predictions", "stds", "stds"));


// See also...

// When we save a model, we must also save the class mappings.  So we use this
// auxiliary structure to store both the perceptron and the mapping, and we'll
// save this.
class RVMRegressionModel
{
 private:
  RVMRegression<> r;
  Col<size_t> map;

 public:
  RVMRegression<>& R() { return r; }
  const RVMRegression<>& R() const { return r; }

  Col<size_t>& Map() { return map; }
  const Col<size_t>& Map() const { return map; }

  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(r);
    ar & BOOST_SERIALIZATION_NVP(map);
  }
};

PARAM_MATRIX_IN("input", "Matrix of covariates (X).", "i");
PARAM_ROW_IN("responses", "Matrix of responses/observations (y).", "r");
PARAM_MODEL_IN(RVMRegressionModel, "input_model", "Trained RVMRegression model to use.", "m");
PARAM_MODEL_OUT(RVMRegressionModel, "output_model", "Output "
                "RVMRegression model.", "M");
PARAM_MATRIX_IN("test", "Matrix containing points to regress on (test "
                "points).", "t");
PARAM_MATRIX_OUT("predictions", "If --test_file is specified, this "
                  "file is where the predicted responses will be saved.", "o");
PARAM_MATRIX_OUT("stds", "If specified, this is where the standard deviations "
    "of the predictive distribution will be saved.", "u");
PARAM_FLAG("center", "Center the data and fit the intercept if enabled.", "c");
PARAM_FLAG("scale", "Scale each feature by their standard deviations if "
           "enabled.", "s");
PARAM_STRING_IN("kernel", "The kernel to use; see the above documentation "
    "for the list of usable kernels.", "k", "");
PARAM_DOUBLE_IN("kernel_scale", "Scale, for 'hyptan' kernel.", "S", 1.0);
PARAM_DOUBLE_IN("offset", "Offset, for 'hyptan' and 'polynomial' kernels.", "O",
    0.0);
PARAM_DOUBLE_IN("bandwidth", "Bandwidth, for 'gaussian' and 'laplacian' "
    "kernels.", "b", 1.0);
PARAM_DOUBLE_IN("degree", "Degree of polynomial, for 'polynomial' kernel.", "D",
    1.0);

//! Run RVMRegression on the specified dataset for the given kernel type.
// template<typename KernelType>
// void RunRVM(const mat& matX,
// 	    const rowvec& responses,
// 	    const KerneType& kernel;
// 	    const bool center,
// 	    const bool scale,
// 	    const bool ard)

// {
  // RVMRegression<KernelType>* rvm;
  // if (input_model)
  // {
  //   rvm = IO::GetParam<RVMRegression<KernelType>*>("inpute_model");
  // }
  // else 
  // {
  //   rvm = new RvMMRegression<KernelType>(kernel, center, scale, ard);
  //   rvm->Train(matX, responses);
  // }
  
  // if (IO::HasParam("test"))
  // {
  //   Log::Info << "Regressing on test points." << endl;
  //   // Load test points.
  //   mat testPoints = std::move(IO::GetParam<mat>("test"));
  //   rowvec predictions;

  //   if (IO::HasParam("stds"))
  //   {
  //     rowvec std;
  //     rvm->Predict(testPoints, predictions, std);

  //     // Save the standard deviation of the test points (one per line).
  //     IO::GetParam<mat>("stds") = std::move(std);
  //   }

  //   else
  //   {
  //     rvm->Predict(testPoints, predictions);
  //   }

  //   // Save test predictions (one per line).
  //   IO::GetParam<mat>("predictions") = std::move(predictions);
  // }

  // IO::GetParam<RVMRegression<KernelType>*>("output_model") = rvm;
// }

static void mlpackMain()
{
  // bool center = IO::GetParam<bool>("center");
  // bool scale = IO::GetParam<bool>("scale");

  // // Check parameters -- make sure everything given make sense.
  // RequireOnlyOnePassed({"input", "input_model"}, true, 
  //     "Pass eihter input data or input model");

  // if (IO::HasParam("input"))
  // {
  //   RequireOnlyOnePassed({"responses"}, true, "if input data is specified, " 
  //       "reponses must also be specified");
  //   mat matX = std::move(IO::GetParam("input"));
  //   rowvec responses = std::move(IO::GetParam("responses"));
  // }

  // ReportIgnoredParam({{"input", false }}, "responses");

  // RequireAtLeastOnePassed({"predictions", "output_model", "stds"}, false, 
  //     "no result will be saved");

  // // Ignore predictions unless test is specified.
  // ReportIgnoredParam({{"test", false}}, "predictions");

  // // If kernel is passed, ensure it is valid.
  // if (IO::HasParam("kernel"))
  // {
  //   // Get the kernel type and make sure it is valid.
  //   RequireParamInSet<string>("kernel", { "linear", "gaussian", "polynomial",
  //       "hyptan", "laplacian", "epanechnikov", "cosine" }, true,
  //       "unknown kernel type");
  //   const string kernelType = IO::GetParam<string>("kernel");
  // }

  // else
  // {
  //   const string kernelType = "ard";
  // }

  // // Instanciation of the estimator according to the kernel specifications.
  // bool ModelPassed = IO::HasParam("input_model");
  // bool ard = false;

  // switch(kernelType)
  // {
  //   case "linear":
  //     LinearKernel kernel;

  //   case "gaussian":
  //     const double bandwidth = IO::GetParam<double>("bandwidth");
  //     GaussianKernel kernel(bandwidth);

  //   case "":
  //     LinearKernel kernel;
  //     const bool ard = true;

  //   default:
  //     std::cout << "Default case, FIX ME" << std::endl;
  // }

  // runRVM(matX, responses, kernel, center, scale, ard)
}

// Il faut faire une fonction template pour l'utilisation du modèle en ligne.
