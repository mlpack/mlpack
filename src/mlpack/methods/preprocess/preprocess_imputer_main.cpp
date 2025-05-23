/**
 * @file methods/preprocess/preprocess_imputer_main.cpp
 * @author Keon Kim
 *
 * A utility that provides imputation strategies for missing values.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME preprocess_imputer

#include <mlpack/core/util/mlpack_main.hpp>

// Program Name.
BINDING_USER_NAME("Impute Data");

// Short description.
BINDING_SHORT_DESC(
    "This utility provides several imputation strategies for missing data. "
    "Given a dataset with missing values, this can impute according to several "
    "strategies, including user-defined values.");

// Long description.
BINDING_LONG_DESC(
    "This utility takes a dataset and converts a user-defined missing variable "
    "to another to provide more meaningful analysis."
    "\n\n"
    "The program does not modify the original matrix, but instead makes a "
    "separate matrix for the output, via the " + PRINT_PARAM_STRING("output") +
    "option.");

// Example.
BINDING_EXAMPLE(
    "For example, if we consider NaN values in dimension 0 to be a missing "
    "variable and want to delete whole data point if it contains a NaN in the "
    "column-wise" + PRINT_DATASET("dataset") + ", we could run:"
    "\n\n" +
    PRINT_CALL("preprocess_imputer", "input", "dataset", "output",
        "result", "dimension", "0", "strategy", "listwise_deletion"));

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
BINDING_SEE_ALSO("@preprocess_split", "#preprocess_split");

PARAM_MATRIX_IN("input", "Input matrix to impute values for.", "i");
PARAM_MATRIX_OUT("output", "Matrix to output that will have imputed values.",
    "o");
PARAM_DOUBLE_IN("missing_value", "Value to use to indicate missing elements "
    "that will be imputed.", "m", std::nan(""));
PARAM_STRING_IN_REQ("strategy", "imputation strategy to be applied. Strategies "
    "should be one of 'custom', 'mean', 'median', and 'listwise_deletion'.",
    "s");
PARAM_DOUBLE_IN("custom_value", "User-defined custom imputation value; only "
    "used if the strategy is 'custom'.", "c", 0.0);
PARAM_INT_IN("dimension", "The dimension to apply imputation to.  If not "
    "specified, missing values will be imputed in every dimension.", "d", 0);

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;
using namespace data;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  mat data = std::move(params.Get<mat>("input"));
  const double missingValue = params.Get<double>("missing_value");
  const double customValue = params.Get<double>("custom_value");
  const size_t dimension = (size_t) params.Get<int>("dimension");
  string strategy = params.Get<string>("strategy");

  RequireParamInSet<string>(params, "strategy", { "custom", "mean", "median",
      "listwise_deletion" }, true, "unknown imputation strategy");
  RequireAtLeastOnePassed(params, { "output" }, false,
      "no output will be saved");

  if (!params.Has("dimension"))
  {
    Log::Info << "--dimension is not specified; the imputation will be "
        << "applied to all dimensions."<< endl;
  }

  if (strategy != "custom")
  {
    ReportIgnoredParam(params, "custom_value", "not using custom imputation "
        "strategy");
  }
  else
  {
    RequireAtLeastOnePassed(params, { "custom_value" }, true, "must pass "
        "custom imputation value when using 'custom' imputation strategy");
  }

  const size_t dimStart = params.Has("dimension") ? dimension : 0;
  const size_t dimEnd = params.Has("dimension") ? dimension + 1 : data.n_rows;

  timers.Start("imputation");

  if (params.Has("dimension"))
  {
    Log::Info << "Performing '" << strategy << "' imputation strategy "
        << "to replace '" << missingValue << "' on dimension " << dimension
        << "." << endl;
  }
  else
  {
    Log::Info << "Performing '" << strategy << "' imputation strategy "
        << "to replace '" << missingValue << "' in all dimensions." << endl;
  }

  for (size_t d = dimStart; d < dimEnd; ++d)
  {
    if (strategy == "mean")
    {
      Imputer<MeanImputation> imputer;
      imputer.Impute(data, missingValue, d);
    }
    else if (strategy == "median")
    {
      Imputer<MedianImputation> imputer;
      imputer.Impute(data, missingValue, d);
    }
    else if (strategy == "listwise_deletion")
    {
      Imputer<ListwiseDeletion> imputer;
      imputer.Impute(data, missingValue, d);
    }
    else if (strategy == "custom")
    {
      CustomImputation<> c(customValue);
      Imputer<CustomImputation<>> imputer(c);
      imputer.Impute(data, missingValue, d);
    }
  }

  params.Get<mat>("output") = std::move(data);
}
