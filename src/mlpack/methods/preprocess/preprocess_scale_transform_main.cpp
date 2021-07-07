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
#define BINDING_NAME preprocess_scale_transform

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
PARAM_MATRIX_OUT("output", "Matrix to save scaled data to.", "o");

PARAM_MODEL_IN_REQ(ScalingModel, "input_model", "Input Scaling model.", "m");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Load the data.
  arma::mat& input = params.Get<arma::mat>("input");
  arma::mat output;
  ScalingModel* m;

  timers.Start("feature_scaling");
  m = params.Get<ScalingModel*>("input_model");
  m->Transform(input, output);
  params.Get<arma::mat>("output") = std::move(output);
  timers.Stop("feature_scaling");
}
