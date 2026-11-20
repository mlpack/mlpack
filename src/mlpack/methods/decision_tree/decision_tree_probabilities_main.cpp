/**
 * @file methods/decision_tree/decision_tree_probabilities_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Implementation of the decision tree classification given a trained model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME decision_tree_probabilities

#include <mlpack/core/util/mlpack_main.hpp>
#include "decision_tree.hpp"
#include "decision_tree_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Decision tree Prediction");

// Short description.
BINDING_SHORT_DESC("Class predictions from train decision tree model.");

// Long description.
BINDING_LONG_DESC("")

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "probabilities", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("Random forest", "#random_forest");
BINDING_SEE_ALSO("Decision trees on Wikipedia",
    "https://en.wikipedia.org/wiki/Decision_tree_learning");
BINDING_SEE_ALSO("Induction of Decision Trees (pdf)",
    "https://www.hunch.net/~coms-4771/quinlan.pdf");
BINDING_SEE_ALSO("DecisionTree C++ class documentation",
    "@doc/user/methods/decision_tree.md");

// Model.
PARAM_MODEL_IN_REQ(DecisionTreeModel, "input_model", "Pre-trained decision "
    "tree, to be used with test points.", "m");

// Datasets.
PARAM_MATRIX_AND_INFO_IN_REQ("test", "Testing dataset (may contain "
    "categorical variables).", "T");
PARAM_UROW_IN("test_labels", "Test point labels, if accuracy calculation "
    "is desired.", "L");

// Output parameters.
PARAM_MATRIX_OUT("probabilities", "Class probabilities for each test point "
    "if probabilities has been selected.", "P");

// Convenience typedef.
using TupleType = tuple<DatasetInfo, arma::mat>;

void BINDING_FUNCTION(util::Params& params, util::Timers& /* timers */)
{
  // Load the model or build the tree.
  DecisionTreeModel* model = params.Get<DecisionTreeModel*>("input_model");

  std::get<0>(params.GetRaw<TupleType>("test")) = model->info;
  arma::mat testPoints = std::get<1>(params.Get<TupleType>("test"));

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  model->tree.Classify(testPoints, predictions, probabilities);

  // Do we need to calculate accuracy?
  if (params.Has("test_labels"))
  {
    arma::Row<size_t> testLabels =
      std::move(params.Get<arma::Row<size_t>>("test_labels"));

    size_t correct = 0;
    for (size_t i = 0; i < testPoints.n_cols; ++i)
      if (predictions[i] == testLabels[i])
        ++correct;

    // Print number of correct points.
    Log::Info << double(correct) / double(testPoints.n_cols) * 100 << "% "
      << "correct on test set (" << correct << " / " << testPoints.n_cols
      << ")." << endl;
  }

  params.Get<arma::mat>("probabilities") = std::move(probabilities);
}
