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

#ifdef BINDING_NAME
  #undef BINDING_NAME
#endif
#define BINDING_NAME preprocess_scale_fit

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
BINDING_USER_NAME("Scale Data");

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
    " The function takes a matrix as ");

// Example.
BINDING_EXAMPLE(
    "So, a simple example where we want to scale the dataset ");

// See also...
BINDING_SEE_ALSO("@preprocess_binarize", "#preprocess_binarize");
BINDING_SEE_ALSO("@preprocess_describe", "#preprocess_describe");
BINDING_SEE_ALSO("@preprocess_imputer", "#preprocess_imputer");

// Define parameters for data.
PARAM_MATRIX_IN_REQ("input", "Matrix containing data.", "i");
PARAM_STRING_IN("scaler_method", "method to use for scaling, the "
    "default is standard_scaler.", "a", "standard_scaler");
PARAM_DOUBLE_IN("epsilon", "regularization Parameter for pcawhitening,"
    " or zcawhitening, should be between -1 to 1.", "r", 0.000001);
PARAM_INT_IN("seed", "Random seed (0 for std::time(NULL)).", "s", 0);
PARAM_INT_IN("min_value", "Starting value of range for min_max_scaler.",
    "b", 0);
PARAM_INT_IN("max_value", "Ending value of range for min_max_scaler.",
    "e", 1);

PARAM_MODEL_OUT(ScalingModel, "output_model", "Output scaling model.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Parse command line options.
  const std::string scalerMethod = params.Get<string>("scaler_method");

  if (params.Get<int>("seed") == 0)
    mlpack::math::RandomSeed(std::time(NULL));
  else
    mlpack::math::RandomSeed((size_t) params.Get<int>("seed"));

  // Check scaler method.
  RequireParamInSet<std::string>(params, "scaler_method", { "min_max_scaler",
    "standard_scaler", "max_abs_scaler", "mean_normalization", "pca_whitening",
    "zca_whitening" }, true, "unknown scaler type");

  // Load the data.
  arma::mat& input = params.Get<arma::mat>("input");

  ScalingModel* m;
  timers.Start("feature_scaling");
  m = new ScalingModel(params.Get<int>("min_value"),
      params.Get<int>("max_value"), params.Get<double>("epsilon"));

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
  params.Get<ScalingModel*>("output_model") = m;
}
