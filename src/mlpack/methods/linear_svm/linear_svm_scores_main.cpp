/**
 * @file methods/linear_svm/linear_svm_scores_main.cpp
 * @author Yashwant Singh Parihar
 * @author Dirk Eddelbuettel
 *
 * Given a trained Linear SVM model, return classification scores from model
 * on new data. The scores are not probabilities (i.e. not bound to [0, 1]).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME linear_svm_scores

#include <mlpack/core/util/mlpack_main.hpp>
#include "linear_svm.hpp"
#include "linear_svm_model.hpp"

using namespace std;
using namespace mlpack;
using namespace mlpack::util;

// Program Name.
BINDING_USER_NAME("Linear SVM Scores");

// Short description.
BINDING_SHORT_DESC("Class scores from Linear SVM model.");

// Long description.
BINDING_LONG_DESC("");

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "scores", "test", "X_test"));

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
PARAM_UROW_IN("test_labels", "Matrix containing test labels.", "L");
PARAM_MATRIX_OUT("scores", "Requested scores.", "p");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // These are the matrices we might use.
  arma::mat trainingSet;
  arma::Row<size_t> labels;
  arma::Row<size_t> rawLabels;
  arma::mat testSet;
  arma::Row<size_t> predictedLabels;

  // Load the model.
  LinearSVMModel* model = params.Get<LinearSVMModel*>("input_model");

  timers.Start("linear_svm_scores");

  // Get the test dataset, and get predictions.
  testSet = std::move(params.Get<arma::mat>("test"));

  arma::Row<size_t> predictions;
  arma::mat scores;
  model->svm.Classify(testSet, predictedLabels, scores);
  RevertLabels(predictedLabels, model->mappings, predictions);

  // Calculate accuracy, if desired.
  if (params.Has("test_labels"))
  {
    arma::Row<size_t> testLabels;
    arma::Row<size_t> testRawLabels =
        std::move(params.Get<arma::Row<size_t>>("test_labels"));

    NormalizeLabels(testRawLabels, testLabels, model->mappings);

    if (testSet.n_cols != testLabels.n_elem)
    {
      Log::Fatal << "Test data given with " << PRINT_PARAM_STRING("test")
          << " has " << testSet.n_cols << " points, but labels in "
          << PRINT_PARAM_STRING("test_labels") << " have "
          << testLabels.n_elem << " labels!" << endl;
    }

    size_t numClasses = model->mappings.n_elem;
    arma::Col<size_t> correctClassCounts;
    arma::Col<size_t> labelSize;
    correctClassCounts.zeros(numClasses);
    labelSize.zeros(numClasses);

    for (arma::uword i = 0; i != predictions.n_elem; ++i)
    {
      if (predictions(i) == testLabels(i))
      {
        ++correctClassCounts[testLabels(i)];
      }
      ++labelSize[testLabels(i)];
    }

    size_t totalCorrectClass = 0;
    for (size_t i = 0; i != correctClassCounts.size(); ++i)
    {
      Log::Info << "Accuracy for points with label " << i << " is "
          << (correctClassCounts[i] / static_cast<double>(labelSize[i]))
          << " (" << correctClassCounts[i] << " of " << labelSize[i] << ")."
          << endl;
      totalCorrectClass += correctClassCounts[i];
    }

    Log::Info << "Total accuracy for all points is "
        << (totalCorrectClass) / static_cast<double>(predictions.n_elem)
        << " (" << totalCorrectClass << " of " << predictions.n_elem << ")."
        << endl;
  }

  timers.Stop("linear_svm_scores");
  params.Get<arma::mat>("scores") = std::move(scores);
}
