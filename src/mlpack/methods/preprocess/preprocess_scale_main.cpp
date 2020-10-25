/**
 * @file methods/preprocess/preprocess_scale_main.cpp
 * @author jeffin sam
 *
 * A binding to scale a dataset.
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
#include <mlpack/core/math/ccov.hpp>
#include <mlpack/core/data/scaler_methods/max_abs_scaler.hpp>
#include <mlpack/core/data/scaler_methods/mean_normalization.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/data/scaler_methods/pca_whitening.hpp>
#include <mlpack/core/data/scaler_methods/zca_whitening.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include "mlpack/methods/preprocess/scaling_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace mlpack::data;
using namespace arma;
using namespace std;

// Program Name.
BINDING_NAME("Scale Data");

// Short description.
BINDING_SHORT_DESC(
    "A utility to perform feature scaling on datasets using one of six"
    "techniques.  Both scaling and inverse scaling are supported, and"
    "scalers can be saved and then applied to other datasets.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes a dataset and performs feature scaling using one of "
    "the six scaler methods namely: 'max_abs_scaler', 'mean_normalization', "
    "'min_max_scaler' ,'standard_scaler', 'pca_whitening' and 'zca_whitening'."
    " The function takes a matrix as " + PRINT_PARAM_STRING("input") +
    " and a scaling method type which you can specify using " +
    PRINT_PARAM_STRING("scaler_method") + " parameter; the default is "
    "standard scaler, and outputs a matrix with scaled feature."
    "\n\n"
    "The output scaled feature matrix may be saved with the " +
    PRINT_PARAM_STRING("output") + " output parameters."
    "\n\n"
    "The model to scale features can be saved using " +
    PRINT_PARAM_STRING("output_model") + " and later can be loaded back using"
    + PRINT_PARAM_STRING("input_model") + ".");

// Example.
BINDING_EXAMPLE(
    "So, a simple example where we want to scale the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_scaled")+ " with "
    " standard_scaler as scaler_method, we could run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "standard_scaler") +
    "\n\n"
    "A simple example where we want to whiten the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_whitened")+ " with "
    " PCA as whitening_method and use 0.01 as regularization parameter, "
    "we could run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "pca_whitening", "epsilon", 0.01) +
    "\n\n"
    "You can also retransform the scaled dataset back using" +
    PRINT_PARAM_STRING("inverse_scaling") + ". An example to rescale : " +
    PRINT_DATASET("X_scaled") + " into " + PRINT_DATASET("X")
    + "using the saved model " + PRINT_PARAM_STRING("input_model") + " is:"
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X_scaled", "output", "X",
    "inverse_scaling", true, "input_model", "saved") +
    "\n\n"
    "Another simple example where we want to scale the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_scaled") + " with "
    " min_max_scaler as scaler method, where scaling range is 1 to 3 instead"
    " of default 0 to 1. We could run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "min_max_scaler", "min_value", 1, "max_value", 3));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Matrix containing data.", "i");
PARAM_MATRIX_OUT("output", "Matrix to save scaled data to.", "o");
PARAM_STRING_IN("scaler_method", "method to use for scaling, the "
    "default is standard_scaler.", "a", "standard_scaler");
PARAM_DOUBLE_IN("epsilon", "regularization Parameter for pcawhitening,"
    " or zcawhitening, should be between -1 to 1.", "r", 0.000001);

PARAM_INT_IN("seed", "Random seed (0 for std::time(NULL)).", "s", 0);
PARAM_INT_IN("min_value", "Starting value of range for min_max_scaler.",
    "b", 0);
PARAM_INT_IN("max_value", "Ending value of range for min_max_scaler.",
    "e", 1);
PARAM_FLAG("inverse_scaling", "Inverse Scaling to get original dataset", "f");
// Loading/saving of a model.
PARAM_MODEL_IN(ScalingModel, "input_model", "Input Scaling model.", "m");
PARAM_MODEL_OUT(ScalingModel, "output_model", "Output scaling model.", "M");

static void mlpackMain()
{
  // Parse command line options.
  const std::string scalerMethod = IO::GetParam<string>("scaler_method");

  if (IO::GetParam<int>("seed") == 0)
    mlpack::math::RandomSeed(std::time(NULL));
  else
    mlpack::math::RandomSeed((size_t) IO::GetParam<int>("seed"));

  // Make sure the user specified output filenames.
  RequireAtLeastOnePassed({ "output", "output_model"}, false,
      "no output will be saved");
  // Check scaler method.
  RequireParamInSet<std::string>("scaler_method", { "min_max_scaler",
    "standard_scaler", "max_abs_scaler", "mean_normalization", "pca_whitening",
    "zca_whitening" }, true, "unknown scaler type");

  // Load the data.
  arma::mat& input = IO::GetParam<arma::mat>("input");
  arma::mat output;
  ScalingModel* m;
  Timer::Start("feature_scaling");
  if (IO::HasParam("input_model"))
  {
    m = IO::GetParam<ScalingModel*>("input_model");
  }
  else
  {
    m = new ScalingModel(IO::GetParam<int>("min_value"),
        IO::GetParam<int>("max_value"), IO::GetParam<double>("epsilon"));

    if (scalerMethod == "standard_scaler")
    {
      m->ScalerType() = ScalingModel::ScalerTypes::STANDARD_SCALER;
    }
    else if (scalerMethod == "min_max_scaler")
    {
      m->ScalerType() = ScalingModel::ScalerTypes::MIN_MAX_SCALER;
    }
    else if (scalerMethod == "max_abs_scaler")
    {
      m->ScalerType() = ScalingModel::ScalerTypes::MAX_ABS_SCALER;
    }
    else if (scalerMethod == "mean_normalization")
    {
      m->ScalerType() = ScalingModel::ScalerTypes::MEAN_NORMALIZATION;
    }
    else if (scalerMethod == "zca_whitening")
    {
      m->ScalerType() = ScalingModel::ScalerTypes::ZCA_WHITENING;
    }
    else if (scalerMethod == "pca_whitening")
    {
      m->ScalerType() = ScalingModel::ScalerTypes::PCA_WHITENING;
    }

    // Fit() can throw an exception on invalid inputs, so we have to catch that
    // and clean the memory in that situation.
    try
    {
      m->Fit(input);
    }
    catch (std::exception& e)
    {
      delete m;
      throw;
    }
  }

  if (!IO::HasParam("inverse_scaling"))
  {
    m->Transform(input, output);
  }
  else
  {
    if (!IO::HasParam("input_model"))
    {
      delete m;
      throw std::runtime_error("Please provide a saved model.");
    }
    m->InverseTransform(input, output);
  }

  // Save the output.
  if (IO::HasParam("output"))
    IO::GetParam<arma::mat>("output") = std::move(output);
  Timer::Stop("feature_scaling");

  IO::GetParam<ScalingModel*>("output_model") = m;
}
