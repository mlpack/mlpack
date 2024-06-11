/**
 * @file methods/xgboost/xgboost_main.cpp
 * @author Abhimanyu Dayal
 *
 * A program to implement XGBoost model. 
 * This file defines the BINDING_FUNCTION()
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

// Defining BINDING_NAME.
#undef BINDING_NAME
#define BINDING_NAME xgboost

// Importing base components required to write mlpack methods.
#include <mlpack/core.hpp>

// Importing checking core utilities (eg. ReportIgnoredParam etc.)
#include <mlpack/core/util/mlpack_main.hpp>

// Importing the BINDING macros.
#include <mlpack/bindings/cli/mlpack_main.hpp>

// Importing xgboost model.
#include "xgboost_model.hpp"

// Defining namespaces.
using namespace std;
using namespace mlpack;
using namespace mlpack::data;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("XGBoost");

// Short description.
BINDING_SHORT_DESC 
(
  "An implementation of XGBoost for classification. " 
  "An xgboost model can be trained and saved; or an "
  "existing xgboost model can be used for classification on new points."
);

// Long description.
BINDING_LONG_DESC 
(
  "PLACEHOLDER TEXT"
)


// Example.
BINDING_EXAMPLE
(
  "PLACEHOLDER TEXT"
)

// See also...
BINDING_SEE_ALSO("XGBoost on Wikipedia", "https://en.wikipedia.org/wiki/"
  "XGBoost");
BINDING_SEE_ALSO("XGBoost: A Scalable Tree Boosting System", 
  "https://arxiv.org/pdf/1603.02754");
BINDING_SEE_ALSO("Decision Stump", "#decision_stump");
BINDING_SEE_ALSO("mlpack::xgboost::XGBoost C++ class documentation",
  "@src/mlpack/methods/xgboost/xgboost.hpp");

// Input for training.
PARAM_MATRIX_IN("training", "Dataset for training XGBoost.", "t");
PARAM_UROW_IN("labels", "Labels for the training set.", "l");

// Classification options.
PARAM_MATRIX_IN("test", "Test dataset.", "T");
PARAM_UROW_OUT("predictions", "Predicted labels for the test set.", "P");
PARAM_MATRIX_OUT("probabilities", "Predicted class probabilities for each "
  "point in the test set.", "p");

// Training parameter options.
// PLACEHOLDER TEXT

// Loading/saving of a model.
PARAM_MODEL_IN(XGBoostModel, "input_model", "Input XGBoost model.", "m");
PARAM_MODEL_OUT(XGBoostModel, "output_model", "Output trained XGBoost model.",
  "M");

// ### DEFINING BINDING FUNCTION

void BINDING_FUNCTION(util::Params& params, util::Timers& timers) 
{
    
}