/**
 * @file preprocess_scale_main.cpp
 * @author jeffin sam
 *
 * A CLI executable to scale a dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/math/random.hpp>
#include <mlpack/core/util/cli.hpp>
#include <mlpack/core/data/scaler_methods/max_abs_scaler.hpp>
#include <mlpack/core/data/scaler_methods/mean_normalization.hpp>
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/data/scaler_methods/pcawhitening.hpp>
#include <mlpack/core/data/scaler_methods/zcawhitening.hpp>
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>
#include "mlpack/core/data/scaler_methods/scaling_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

PROGRAM_INFO("Scale Data",
    // Short description.
    "A utility to perform feature scaling.",
    // Long description.
    "This utility takes a dataset and performs feature scaling using one of "
    "the six scaler methods namely: 'max_abs_scaler', 'mean_normalization', "
    "'min_max_scaler' ,'standard_scaler', 'pcawhitening' and 'zcawhitening'."
    " The function takes a matrice as " + PRINT_PARAM_STRING("input") +
    " and a scaling method type which you can specify using " +
    PRINT_PARAM_STRING("scaler_method") + " parameter; the default is "
    "standard scaler, and outputs a matrice with scaled feature."
    "\n\n"
    "The output scaled feature matrices may be saved with the " +
    PRINT_PARAM_STRING("output") + " output parameters."
    "\n\n"
    "The model to scale features can be saved using " +
    PRINT_PARAM_STRING("output_model") + " and later can be loaded back using"
    + PRINT_PARAM_STRING("input_model") + "."
    "\n\n"
    "So, a simple example where we want to scale the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_scaled")+ " with "
    " standard_scaler as scaler_method, we coud run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "standard_scaler") +
    "\n\n"
    "A simple example where we want to whiten the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_whitened")+ " with "
    " PCA as whitening_method and use 0.01 as regularization parameter, "
    "we coud run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "pcawhitening", "epsilon", 0.01) +
    "\n\n"
    "You can also retransform the scaled dataset back using" +
    PRINT_PARAM_STRING("function") + ". An example to rescale : " +
    PRINT_DATASET("X_scaled") + " into " + PRINT_DATASET("X")
    + "using the saved model " + PRINT_PARAM_STRING("input_model") + " is:"
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X_scaled", "output", "X",
    "scaler_method", "standard_scaler", "function", 1, "input_model", "saved")+
    "\n\n"
    "Another simple example where we want to scale the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_scaled") + " with "
    " min_max_scaler as scaler method, where scaling range is 1 to 3 instead"
    " of deafult 0 to 1. We coud run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "min_max_scaler", "min_value", 1, "max_value", 3),
    SEE_ALSO("@preprocess_binarize", "#preprocess_binarize"),
    SEE_ALSO("@preprocess_describe", "#preprocess_describe"),
    SEE_ALSO("@preprocess_imputer", "#preprocess_imputer"));

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
PARAM_INT_IN("function", "function to apply,either 0:Transform or 1:Inverse"
  "Transform", "f", 0)
// Loading/saving of a model.
PARAM_MODEL_IN(data::ScalingModel, "input_model", "Input Scaling model.", "m");
PARAM_MODEL_OUT(data::ScalingModel, "output_model", "Output scaling model.",
    "M");

static void mlpackMain()
{
  // Parse command line options.
  const std::string scalerMethod = CLI::GetParam<string>("scaler_method");

  if (CLI::GetParam<int>("seed") == 0)
    mlpack::math::RandomSeed(std::time(NULL));
  else
    mlpack::math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  // Make sure the user specified output filenames.
  RequireAtLeastOnePassed({ "output", "output_model"}, false,
      "no output will be saved");
  // Check scaler method.
  RequireParamValue<std::string>("scaler_method",
      [](std::string x) { return x == "standard_scaler" || x ==
      "min_max_scaler" || x == "mean_normalization" || x == "max_abs_scaler" ||
      x == "whitening";}, true, "scaler_method must be one among min_max_"
      "scaler, max_abs_scaler, pcawhitening, standard_scaler, zcawhitening"
      " or mean_normalization.");

  // Load the data.
  arma::mat& input = CLI::GetParam<arma::mat>("input");
  arma::mat output;
  data::ScalingModel* m;
  if (CLI::HasParam("input_model"))
  {
    m = CLI::GetParam<data::ScalingModel*>("input_model");
  }
  else
  {
    m = new data::ScalingModel(CLI::GetParam<int>("min_value"),
        CLI::GetParam<int>("max_value"), CLI::GetParam<double>("epsilon"));
    if (scalerMethod == "standard_scaler")
    {
      m->ScalerType() = data::ScalingModel::ScalerTypes::STANDARD_SCALER;
    }
    else if (scalerMethod == "min_max_scaler")
    {
      m->ScalerType() = data::ScalingModel::ScalerTypes::MIN_MAX_SCALER;
    }
    else if (scalerMethod == "max_abs_scaler")
    {
      m->ScalerType() = data::ScalingModel::ScalerTypes::MAX_ABS_SCALER;
    }
    else if (scalerMethod == "mean_normalization")
    {
      m->ScalerType() = data::ScalingModel::ScalerTypes::MEAN_NORMALIZATION;
    }
    else if (scalerMethod == "zcawhitening")
    {
      m->ScalerType() = data::ScalingModel::ScalerTypes::ZCAWHITENING;
    }
    else if (scalerMethod == "pcawhitening")
    {
      m->ScalerType() = data::ScalingModel::ScalerTypes::PCAWHITENING;
    }
    m->Fit(input);
  }
  if (!CLI::GetParam<int>("function"))
  {
    m->Transform(input, output);
  }
  else
  {
    m->InverseTransform(input, output);
  }

  // save the output
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(output);

  CLI::GetParam<data::ScalingModel*>("output_model") = m;
}
