/**
 * @file methods/random_forest/random_forest_classify_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Given a trained random forest model, predict from model on new data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME random_forest_classify

#include <mlpack/core/util/mlpack_main.hpp>
#include "random_forest.hpp"
#include "random_forest_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_USER_NAME("Random Forests Prediction");

// Short description.
BINDING_SHORT_DESC("Class predictions from random forest model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "classify", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@hoeffding_tree", "#hoeffding_tree");
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("Random forest on Wikipedia",
    "https://en.wikipedia.org/wiki/Random_forest");
BINDING_SEE_ALSO("Random forests (pdf)", "https://www.eecis.udel.edu/~shatkay"
    "/Course/papers/BreimanRandomForests2001.pdf");
BINDING_SEE_ALSO("RandomForest C++ class documentation",
    "@doc/user/methods/random_forest.md");

PARAM_MODEL_IN_REQ(RandomForestModel, "input_model", "Pre-trained random "
    "forest to use for classification.", "m");

PARAM_MATRIX_IN_REQ("test", "Test dataset to produce predictions for.", "T");
PARAM_UROW_IN("test_labels", "Test dataset labels, if accuracy calculation is "
    "desired.", "L");

PARAM_UROW_OUT("predictions", "Predicted classes for each point in the test "
    "set.", "p");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  RandomForestModel* rfModel = params.Get<RandomForestModel*>("input_model");

  arma::mat testData = std::move(params.Get<arma::mat>("test"));
  timers.Start("rf_prediction");

  // Get predictions.
  arma::Row<size_t> predictions;
  rfModel->rf.Classify(testData, predictions);

  // Did we want to calculate test accuracy?
  if (params.Has("test_labels"))
  {
    arma::Row<size_t> testLabels =
        std::move(params.Get<arma::Row<size_t>>("test_labels"));

    const size_t correct = accu(predictions == testLabels);

    Log::Info << correct << " of " << testLabels.n_elem << " correct on test"
        << " set (" << (double(correct) / double(testLabels.n_elem) * 100)
        << ")." << endl;
    timers.Stop("rf_prediction");
  }

  // Save the outputs.
  params.Get<arma::Row<size_t>>("predictions") = std::move(predictions);
}
