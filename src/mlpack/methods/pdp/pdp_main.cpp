/**
 * @file methods/pdp/pdp_main.cpp
 * @author Ankit Singh
 *
 * Main function for Partial Dependence Plot (PDP) computation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include "pdp.hpp"

#undef BINDING_NAME
#define BINDING_NAME pdp

#include <mlpack/core/util/mlpack_main.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace arma;
using namespace std;

// Program Name.
BINDING_USER_NAME("Partial Dependence Plot (PDP) Computation");

// Short description.
BINDING_SHORT_DESC("Compute Partial Dependence Plots (PDP) for mlpack models.");

// Long description.
BINDING_LONG_DESC(
    "This program computes the Partial Dependence Plot (PDP) for a given "
    "machine learning model by varying a specific feature across its range "
    "while keeping other features constant, and averaging predictions. "
    "The results can be used to interpret model behavior.");

// Example usage.
BINDING_EXAMPLE(
    "For example, to compute the PDP for feature index 2 on a trained model "
    "using dataset " + PRINT_DATASET("data") + ", saving output to " + PRINT_DATASET("pdp_output") + " the following command could be used:"
    "\n\n" +
    PRINT_CALL("pdp", "input_model", "trained_model", "data", "data", "feature_index", "2", "num_points", "50", "output", "pdp_output"));

// Parameters.
PARAM_MODEL_IN(ModelType, "input_model", "Trained machine learning model.", "m");
PARAM_MATRIX_IN("data", "Dataset on which to compute PDP.", "d");
PARAM_INT_IN("feature_index", "Index of the feature for PDP computation.", "f", 0);
PARAM_INT_IN("num_points", "Number of points for feature variation.", "n", 50);
PARAM_MATRIX_OUT("output", "Output matrix containing computed PDP values.", "o");

void BINDING_FUNCTION(util::Params& params, util::Timers& timer)
{
    // Load parameters.
    ModelType* model = params.Get<ModelType*>("input_model");
    mat data = params.Get<mat>("data");
    size_t featureIndex = (size_t)params.Get<int>("feature_index");
    size_t numPoints = (size_t)params.Get<int>("num_points");

    // Ensure valid feature index.
    if (featureIndex >= data.n_rows)
    {
        Log::Fatal << "Feature index out of bounds!" << endl;
    }

    // Compute PDP.
    timer.Start("pdp_computation");
    PDP<ModelType, Policy> pdp(*model, data, featureIndex, numPoints);
    arma::vec featureValues, pdpValues;
    std::tie(featureValues, pdpValues) = pdp.Compute();
    timer.Stop("pdp_computation");

    // Combine feature values and PDP values into output matrix.
    mat output(2, numPoints);
    output.row(0) = featureValues.t();
    output.row(1) = pdpValues.t();

    // Save output.
    params.Get<mat>("output") = std::move(output);
}
