/**
 * @file methods/preprocess/preprocess_binarize_main.cpp
 * @author Keon Kim
 *
 * A binding to binarize a dataset.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME preprocess_binarize

#include <mlpack/core/util/mlpack_main.hpp>

// Program Name.
BINDING_USER_NAME("Binarize Data");

// Short description.
BINDING_SHORT_DESC(
    "A utility to binarize a dataset.  Given a dataset, this utility converts "
    "each value in the desired dimension(s) to 0 or 1; this can be a useful "
    "preprocessing step.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes a dataset and binarizes the "
    "variables into either 0 or 1 given threshold. User can apply binarization "
    "on a dimension or the whole dataset.  The dimension to apply binarization "
    "to can be specified using the " + PRINT_PARAM_STRING("dimension") +
    " parameter; if left unspecified, every dimension will be binarized.  The "
    "threshold for binarization can also be specified with the " +
    PRINT_PARAM_STRING("threshold") + " parameter; the default threshold is "
    "0.0."
    "\n\n"
    "The binarized matrix may be saved with the " +
    PRINT_PARAM_STRING("output") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "For example, if we want to set all variables greater than 5 in the "
    "dataset " + PRINT_DATASET("X") + " to 1 and variables less than or equal "
    "to 5.0 to 0, and save the result to " + PRINT_DATASET("Y") + ", we could "
    "run"
    "\n\n" +
    PRINT_CALL("preprocess_binarize", "input", "X", "threshold", 5.0, "output",
        "Y") +
    "\n\n"
    "But if we want to apply this to only the first (0th) dimension of " +
    PRINT_DATASET("X") + ",  we could instead run"
    "\n\n" +
    PRINT_CALL("preprocess_binarize", "input", "X", "threshold", 5.0,
        "dimension", 0, "output", "Y"));

// See also...
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
#if BINDING_TYPE == BINDING_TYPE_CLI
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");
#endif
BINDING_SEE_ALSO("@preprocess_split", "#preprocess_split");

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Input data matrix.", "i");
// Define optional parameters.
PARAM_MATRIX_OUT("output", "Matrix in which to save the output.", "o");
PARAM_INT_IN("dimension", "Dimension to apply the binarization. If not set, the"
    " program will binarize every dimension by default.", "d", 0);
PARAM_DOUBLE_IN("threshold", "Threshold to be applied for binarization. If not "
    "set, the threshold defaults to 0.0.", "t", 0.0);

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  const size_t dimension = (size_t) params.Get<int>("dimension");
  const double threshold = params.Get<double>("threshold");

  // Check on data parameters.
  if (!params.Has("dimension"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("dimension")
        << ", so the program will perform binarization on every dimension."
        << endl;
  }

  if (!params.Has("threshold"))
  {
    Log::Warn << "You did not specify " << PRINT_PARAM_STRING("threshold")
        << ", so the threshold will be automatically set to '0.0'." << endl;
  }

  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  // Load the data.
  arma::mat input = std::move(params.Get<arma::mat>("input"));
  arma::mat output;

  RequireParamValue<int>(params, "dimension", [](int x) { return x >= 0; },
      true, "dimension to binarize must be nonnegative");
  std::ostringstream error;
  error << "dimension to binarize must be less than the number of dimensions "
      << "of the input data (" << input.n_rows << ")";
  RequireParamValue<int>(params, "dimension",
      [input](int x) { return size_t(x) < input.n_rows; }, true, error.str());

  timers.Start("binarize");
  if (params.Has("dimension"))
  {
    data::Binarize<double>(input, output, threshold, dimension);
  }
  else
  {
    // Binarize the whole dataset.
    data::Binarize<double>(input, output, threshold);
  }
  timers.Stop("binarize");

  if (params.Has("output"))
    params.Get<arma::mat>("output") = std::move(output);
}
