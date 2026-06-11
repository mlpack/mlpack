/**
 * @file methods/linear_svm/linear_svm_probabilities_main.cpp
 * @author Yashwant Singh Parihar
 * @author Dirk Eddelbuettel
 *
 * Given a trained Linear SVM model, classify from model on new data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME linear_svm_probabilities

#include <mlpack/core/util/mlpack_main.hpp>
#include "linear_svm.hpp"
#include "linear_svm_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Linear SVM Probabilities");

// Short description.
BINDING_SHORT_DESC("Class probabilities from Linear SVM model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "probabilities", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("@logistic_regression", "#logistic_regression");
BINDING_SEE_ALSO("LinearSVM on Wikipedia",
    "https://en.wikipedia.org/wiki/Support-vector_machine");
BINDING_SEE_ALSO("LinearSVM C++ class documentation",
    "@doc/user/methods/linear_svm.md");

// Model loading
PARAM_MODEL_IN_REQ(LinearSVMModel, "input_model", "Existing model "
    "(parameters).", "m");
// Testing.
PARAM_MATRIX_IN_REQ("test", "Matrix containing test dataset.", "T");
PARAM_MATRIX_OUT("probabilities", "Requested probabilities.", "p");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // These are the matrices we might use.
  arma::mat trainingSet;
  arma::Row<size_t> labels;
  arma::Row<size_t> rawLabels;
  arma::mat testSet;
  arma::Row<size_t> predictedLabels;
  size_t numClasses;

  // Load the model.
  LinearSVMModel* model = params.Get<LinearSVMModel*>("input_model");

  timers.Start("linear_svm_probabilities");

  // Get the test dataset, and get predictions.
  testSet = std::move(params.Get<arma::mat>("test"));

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  model->svm.Classify(testSet, predictions, probabilities);
  params.Get<arma::mat>("probabilities") = std::move(probabilities);
  model->svm.Classify(testSet, predictedLabels);
  RevertLabels(predictedLabels, model->mappings, predictions);

  timers.Stop("linear_svm_probabilities");
}
