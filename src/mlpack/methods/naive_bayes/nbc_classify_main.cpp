/**
 * @file methods/naive_bayes/nbc_classify_main.cpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * author Dirk Eddelbuettel
 *
 * Given a train Naive Bayes Classifier, predict from the model on new data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME nbc_classify

#include <mlpack/core/util/mlpack_main.hpp>

#include "naive_bayes_classifier.hpp"
#include "naive_bayes_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Parametric Naive Bayes Classifier Prediction");

// Short description.
BINDING_SHORT_DESC("Class predictions from a Naive Bayes Classifier model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "classify", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Naive Bayes classifier on Wikipedia",
    "https://en.wikipedia.org/wiki/Naive_Bayes_classifier");
BINDING_SEE_ALSO("NaiveBayesClassifier C++ class documentation",
    "@doc/user/methods/naive_bayes_classifier.md");

// Model loading.
PARAM_MODEL_IN_REQ(NBCModel, "input_model", "Input Naive Bayes "
    "model.", "m");

// Test parameters.
PARAM_MATRIX_IN_REQ("test", "A matrix containing the test set.", "T");
PARAM_UROW_OUT("predictions", "The matrix in which the predicted labels for the"
    " test set will be written.", "a");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Load a model and load test data.
  NBCModel* model = params.Get<NBCModel*>("input_model");
  mat testingData = std::move(params.Get<mat>("test"));

  if (testingData.n_rows != model->nbc.Means().n_rows)
  {
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << model->nbc.Means().n_rows
        << ")!" << std::endl;
  }

  // Time the running of the Naive Bayes Classifier.
  Row<size_t> predictions;
  timers.Start("nbc_prediction");
  model->nbc.Classify(testingData, predictions);
  timers.Stop("nbc_prediction");

  // Un-normalize labels to prepare output.
  Row<size_t> rawResults;
  RevertLabels(predictions, model->mappings, rawResults);
  params.Get<Row<size_t>>("predictions") = std::move(rawResults);
}
