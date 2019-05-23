/**
 * @file preprocess_scale_main.cpp
 * @author jeffin sam
 *
 * A CLI executable to split a dataset.
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
#include <mlpack/core/data/scaler_methods/standard_scaler.hpp>

PROGRAM_INFO("Scale Data",
    // Short description.
    "A utility to perform feature scaling.",
    // Long description.
    "This utility takes a dataset and performs feature scaling using one of "
    "the four scaler methods namely: max_abs_scaler, mean_normalization, "
    "min_max_scaler and standard_scaler. The function takes a matrice as " +
    PRINT_PARAM_STRING("input") + " and a scaling method type which "
    "you can specify using " + PRINT_PARAM_STRING("scaler_method") +
    " parameter; the default is standard scaler, and outputs a matrice "
    "with scaled feature."
    "\n\n"
    "The output scaled feature matrices may be saved with the " +
    PRINT_PARAM_STRING("output") + " output parameters."
    "\n\n"
    "So, a simple example where we want to scale the dataset " +
    PRINT_DATASET("X") + " into " + PRINT_DATASET("X_scaled")+ " with "
    " standard_scaler as scaler_method, we coud run "
    "\n\n" +
    PRINT_CALL("preprocess_scale", "input", "X", "output", "X_scaled",
    "scaler_method", "standard_scaler") +
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

PARAM_INT_IN("seed", "Random seed (0 for std::time(NULL)).", "s", 0);
PARAM_INT_IN("min_value", "Starting value of range for min_max_scaler.",
    "b", 0);
PARAM_INT_IN("max_value", "Ending value of range for min_max_scaler.",
    "e", 1);

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

static void mlpackMain()
{
  // Parse command line options.
  const std::string scalerMethod = CLI::GetParam<string>("scaler_method");

  if (CLI::GetParam<int>("seed") == 0)
    mlpack::math::RandomSeed(std::time(NULL));
  else
    mlpack::math::RandomSeed((size_t) CLI::GetParam<int>("seed"));
  // Make sure the user specified output filenames.
  RequireAtLeastOnePassed({ "output" }, false, "no output will be saved");
  // Check scaler method.
  RequireParamValue<std::string>("scaler_method",
      [](std::string x) { return x == "standard_scaler" || x == 
      "min_max_scaler" || x == "mean_normalization" || x == "max_abs_scaler";},
      true, "scaler_method must be one among standard_scaler, max_abs_scaler,"
      "min_max_scaler or mean_normalization.");
  // If scaler_method is not set, warn the user.
  if (!CLI::HasParam("scaler_method"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("scaler_method")
        << ", so it will be automatically set to standard_scaler." << endl;
  }

  // Load the data.
  arma::mat input = CLI::GetParam<arma::mat>("input");
  arma::mat output;
  if (scalerMethod == "min_max_scaler")
  {
    if (!CLI::HasParam("min_value"))
    {
      Log::Warn << "You did not specify " << PRINT_PARAM_STRING("min_value")
          << ", so it will be automatically set to 0." << endl;
    }
    if (!CLI::HasParam("max_value"))
    {
      Log::Warn << "You did not specify " << PRINT_PARAM_STRING("max_value")
          << ", so it will be automatically set to 1." << endl;
    }
    const int minValue = CLI::GetParam<int>("min_value");
    const int maxValue = CLI::GetParam<int>("max_value");
    data::MinMaxScaler scale(minValue, maxValue);
    scale.Transform(input, output);
  }
  else if (scalerMethod == "max_abs_scaler")
  {
    data::MaxAbsScaler scale;
    scale.Transform(input, output);
  }
  else if (scalerMethod == "mean_normalization")
  {
    data::MeanNormalization scale;
    scale.Transform(input, output);
  }
  else
  {
    data::StandardScaler scale;
    scale.Transform(input, output);
  }
  // save the output
  if (CLI::HasParam("output"))
    CLI::GetParam<arma::mat>("output") = std::move(output);
}
