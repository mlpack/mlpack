/**
 * @file methods/hoeffding_trees/hoeffding_tree_probabilities_main.cpp
 * @author Ryan Curtin
 * @author Dirk Eddelbuettel
 *
 * Given a trained Hoeffding trees model, return class probailities from
 * the model on new data.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME hoeffding_tree_probabilities

#include <mlpack/core/util/mlpack_main.hpp>

#include <mlpack/methods/hoeffding_trees/hoeffding_tree.hpp>
#include <mlpack/methods/hoeffding_trees/binary_numeric_split.hpp>
#include <mlpack/methods/hoeffding_trees/information_gain.hpp>
#include <mlpack/methods/hoeffding_trees/hoeffding_tree_model.hpp>

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Hoeffding trees Probabilities");

// Short description.
BINDING_SHORT_DESC("Class probabilities from Hoeffding trees model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "probabilities", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@decision_tree", "#decision_tree");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Mining High-Speed Data Streams (pdf)",
    "https://www.cs.rhodes.edu/~welshc/COMP465_S15/Papers/kdd00.pdf");
BINDING_SEE_ALSO("HoeffdingTree class documentation",
    "@doc/user/methods/hoeffding_tree.md");

PARAM_MODEL_IN_REQ(HoeffdingTreeModel, "input_model",
    "Input trained Hoeffding tree model.", "m");

PARAM_MATRIX_AND_INFO_IN_REQ("test",
    "Testing dataset (may be categorical).", "T");
PARAM_UROW_IN("test_labels", "Labels of test data.", "L");
PARAM_MATRIX_OUT("probabilities", "In addition to predicting labels, provide "
    "rediction probabilities in this matrix.", "P");

// Convenience typedef.
using TupleType = tuple<DatasetInfo, arma::mat>;

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  HoeffdingTreeModel* model = params.Get<HoeffdingTreeModel*>("input_model");
  DatasetInfo datasetInfo;

  // Before loading, pre-set the dataset info by getting the raw parameter
  // (that doesn't call Load()).
  std::get<0>(params.GetRaw<TupleType>("test")) = datasetInfo;
  arma::mat testSet = std::get<1>(params.Get<TupleType>("test"));

  arma::Row<size_t> predictions;
  arma::rowvec probabilities;

  timers.Start("tree_testing");
  model->Classify(testSet, predictions, probabilities);
  timers.Stop("tree_testing");

  if (params.Has("test_labels"))
  {
    arma::Row<size_t> testLabels =
      std::move(params.Get<arma::Row<size_t>>("test_labels"));

    size_t correct = 0;
    for (size_t i = 0; i < testLabels.n_elem; ++i)
    {
      if (predictions[i] == testLabels[i])
        ++correct;
    }
    Log::Info << correct << " out of " << testLabels.n_elem << " correct "
        << "on test set (" << double(correct) / double(testLabels.n_elem) *
        100.0 << ")." << endl;
  }

  params.Get<arma::mat>("probabilities") = std::move(probabilities);
}
