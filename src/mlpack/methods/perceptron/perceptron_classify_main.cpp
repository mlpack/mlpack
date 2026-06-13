/**
 * @file methods/perceptron/perceptron_classify_main.cpp
 * @author Udit Saxena
 * @author Dirk Eddelbuettel
 *
 * Implementation of the simple perceptron classification given a trained model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME perceptron_classify

#include <mlpack/core/util/mlpack_main.hpp>

#include "perceptron.hpp"
#include "perceptron_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Perceptron Prediction");

// Short description.
BINDING_SHORT_DESC("Class predictions from perceptron model.");

// Long description.
BINDING_LONG_DESC("")

// Example.
BINDING_EXAMPLE(
    CALL_METHOD("model", "classify", "test", "X_test"));

// See also...
BINDING_SEE_ALSO("@adaboost", "#adaboost");
BINDING_SEE_ALSO("Perceptron on Wikipedia",
    "https://en.wikipedia.org/wiki/Perceptron");
BINDING_SEE_ALSO("Perceptron C++ class documentation",
    "@doc/user/methods/perceptron.md");

// Model loading.
PARAM_MODEL_IN_REQ(PerceptronModel, "input_model", "Input perceptron model.",
    "m");

// Testing/classification parameters.
PARAM_MATRIX_IN_REQ("test", "A matrix containing the test set.", "T");
PARAM_UROW_OUT("predictions", "The matrix in which the predicted labels for the"
    " test set will be written.", "P");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // Now, load our model, if there is one.
  PerceptronModel* p = params.Get<PerceptronModel*>("input_model");

  mat& testData = params.Get<arma::mat>("test");
  Log::Info << "Classifying dataset '"
      << params.GetPrintable<arma::mat>("test") << "'." << endl;

  if (testData.n_rows != p->P().Weights().n_rows)
  {
    const size_t perceptronDimensionality = p->P().Weights().n_rows;
    Log::Fatal << "Test data dimensionality (" << testData.n_rows << ") must "
        << "be the same as the dimensionality of the perceptron ("
        << perceptronDimensionality << ")!" << endl;
  }

  // Time the running of the perceptron classifier.
  Row<size_t> predictedLabels(testData.n_cols);
  timers.Start("testing");
  p->P().Classify(testData, predictedLabels);
  timers.Stop("testing");

  // Un-normalize labels to prepare output.
  Row<size_t> results;
  RevertLabels(predictedLabels, p->Map(), results);

  // Save the predicted labels.
  params.Get<arma::Row<size_t>>("predictions") = std::move(results);
}
