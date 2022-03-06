/**
 * @file rvm_regression_main.cpp
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

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME rvm_regression

#include <mlpack/core/util/mlpack_main.hpp>

#include "rvm_regression.hpp"
#include "rvm_regression_model.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::util;
using namespace mlpack::kernel;

// Program Name.
BINDING_USER_NAME("Relevance Vector Machine for regression");

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
    PRINT_PARAM_STRING("stds") + " parameter. "
    "If the " + PRINT_PARAM_STRING("kernel") + " is not specified the model "
    "optimized is a bayesian linear regression whose the solution is associa- "
    "ted to an ARD prior leading to sparse solution in the features domain."
    "\n"
    "The supported kernel are listed below:"
    "\n\n"
    " * 'linear': the standard linear dot product (same as normal PCA):\n"
    "    K(x, y) = x^T y\n"
    "\n"
    " * 'gaussian': a Gaussian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y || ^ 2) / (2 * (bandwidth ^ 2)))\n"
    "\n"
    " * 'polynomial': Polynomial kernel; requires offset and degree:\n"
    "    K(x, y) = (x^T y + offset) ^ degree\n"
    "\n"
    // " * 'hyptan': hyperbolic tangent kernel; requires scale and offset:\n"
    // "    K(x, y) = tanh(scale * (x^T y) + offset)\n"
    // "\n"
    " * 'laplacian': Laplacian kernel; requires bandwidth:\n"
    "    K(x, y) = exp(-(|| x - y ||) / bandwidth)\n"
    "\n"
    " * 'epanechnikov': Epanechnikov kernel; requires bandwidth:\n"
    "    K(x, y) = max(0, 1 - || x - y ||^2 / bandwidth^2)\n"
    "\n"
    " * 'cosine': Cosine distance:\n"
    "    K(x, y) = 1 - (x^T y) / (|| x || * || y ||)\n"
    "\n"
    " * 'sperical': Spherical kernel; requires bandwidth:\n"
    "                                                \n"
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


PARAM_MATRIX_IN("input", "Matrix of covariates (X).", "i");
PARAM_ROW_IN("responses", "Matrix of responses/observations (y).", "r");
PARAM_MODEL_IN(RVMRegressionModel, "input_model", "Trained RVMRegression model "
	       " to use.", "m");
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
PARAM_DOUBLE_IN("bandwidth", "Bandwidth, for 'gaussian', 'laplacian', "
		"'spherical' and 'epanechnikov' kernels.", "b", 1.0);
PARAM_STRING_IN("kernel", "The kernel to use; see the above documentation "
    "for the list of usable kernels.", "k", "");
PARAM_DOUBLE_IN("kernel_scale", "Scale, for 'hyptan' kernel.", "S", 1.0);
PARAM_DOUBLE_IN("offset", "Offset, for 'hyptan' and 'polynomial' kernels.", "O",
    0.0);
PARAM_DOUBLE_IN("degree", "Degree of polynomial, for 'polynomial' kernel.", "D",
    2.0);


void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  bool center = params.Get<bool>("center");
  bool scale = params.Get<bool>("scale");

  // Check parameters -- make sure everything given make sense.
  RequireOnlyOnePassed(params, {"input", "input_model"}, true, 
      "Pass eihter input data or input model");

  mat matX;
  rowvec responses;
  if (params.Has("input"))
  {
    RequireOnlyOnePassed(params, {"responses"}, true,
        "if input data is specified, reponses must also be specified");
    matX = std::move(params.Get<arma::mat>("input"));
    responses = std::move(params.Get<arma::rowvec>("responses"));

    if (responses.n_elem != matX.n_cols)
    {
      Log::Fatal << "Number of responses must be equal to number of rows of X!"
                 << endl;
    }
  }

  ReportIgnoredParam(params, {{"input", false }}, "responses");

  RequireAtLeastOnePassed(params, {"predictions", "output_model", "stds"},
      false, "no result will be saved");

  // Ignore predictions unless test is specified.
  ReportIgnoredParam(params, {{"test", false}}, "predictions");

  // If kernel is passed, ensure it is valid.
  string kernelType;
  if (params.Has("kernel"))
  {
    // Get the kernel type and make sure it is valid.
    RequireParamInSet<string>(params, "kernel",
			      { "linear", "gaussian", "polynomial",
				"hyptan", "laplacian", "epanechnikov",
				"cosine", "spherical" },
			      true, "unknown kernel type");
    
    kernelType = params.Get<string>("kernel");
  }
  else
  {
    kernelType = "ard";
  }

  RVMRegressionModel* estimator;
  
  if (params.Has("input_model"))
  {
    estimator = params.Get<RVMRegressionModel*>("input_model");
  }
  else 
  {
    // Create and train the RVM.
    estimator = new RVMRegressionModel(kernelType, center, scale,
				       params.Get<double>("bandwidth"),
				       params.Get<double>("offset"),
				       params.Get<double>("kernel_scale"),
				       params.Get<double>("degree"));
    estimator->Train(matX, responses);
  }
  
  if (params.Has("test"))
  {
    Log::Info << "Regressing on test points." << endl;
    // Load test points.
    mat testPoints = std::move(params.Get<mat>("test"));
    rowvec predictions;

    if (params.Has("stds"))
    {
      Log::Info << "Uncertainties computed." << endl;
      rowvec std;
      estimator->Predict(testPoints, predictions, std);

      // Save the standard deviation of the test points (one per line).
      params.Get<mat>("stds") = std::move(std);
    }
    else
    {
      estimator->Predict(testPoints, predictions);
    }

    // Save test predictions (one per line).
    params.Get<mat>("predictions") = std::move(predictions);
  }

  params.Get<RVMRegressionModel*>("output_model") = estimator;
}

