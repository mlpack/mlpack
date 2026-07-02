/**
 * @file methods/perceptron/perceptron_train_main.cpp
 * @author Udit Saxena
 * @author Dirk Eddelbuettel
 *
 * Implementation of the Simple Perceptron Classifier training.
 *
 * Perceptrons are simple single-layer binary classifiers, which solve linearly
 * separable problems with a linear decision boundary.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#undef BINDING_NAME
#define BINDING_NAME perceptron_train

#include <mlpack/core/util/mlpack_main.hpp>
#include "perceptron.hpp"
#include "perceptron_model.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Perceptron training");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of a perceptron---a single level neural network---for "
    "classification.  Given labeled data, a perceptron can be trained and "
    "later be used for classification on new points.");

// Long description.
BINDING_LONG_DESC(
    "Implementation of a perceptron, which is a single level neural "
    "network. The perceptron makes its predictions based on a linear predictor "
    "function combining a set of weights with the feature vector.  The "
    "perceptron learning rule is able to converge, given enough iterations "
    "(specified using the " + PRINT_PARAM_STRING("max_iterations") +
    " parameter), if the data supplied is linearly separable.  The perceptron "
    "is parameterized by a matrix of weight vectors that denote the numerical "
    "weights of the neural network.");

// Example.
BINDING_EXAMPLE(
    IMPORT_EXT_LIB() + "\n" +
    IMPORT_SPLIT() + "\n" +
    IMPORT_THIS("perceptron", "train", "classify") + "\n" +
    GET_DATASET("X", "http://datasets.mlpack.org/iris.csv") + "\n" +
    GET_DATASET("y", "http://datasets.mlpack.org/iris_labels.csv") + "\n" +
    SPLIT_TRAIN_TEST("X", "y", "X_train", "y_train", "X_test", "y_test",
    "0.2") + "\n" +
    CREATE_OBJECT("model", "perceptron") + "\n" +
    CALL_METHOD("model", "train", "training", "X_train", "labels", "y_train",
                "max_iterations", 100));

// See also...
BINDING_SEE_ALSO("@adaboost", "#adaboost");
BINDING_SEE_ALSO("Perceptron on Wikipedia",
    "https://en.wikipedia.org/wiki/Perceptron");
BINDING_SEE_ALSO("Perceptron C++ class documentation",
    "@doc/user/methods/perceptron.md");

// Training parameters.
PARAM_MATRIX_IN_REQ("training", "A matrix containing the training set.", "t");
PARAM_UROW_IN("labels", "A matrix containing labels for the training set.",
    "l");
PARAM_INT_IN("max_iterations", "The maximum number of iterations the "
    "perceptron is to be run", "n", 1000);

PARAM_MODEL_OUT(PerceptronModel, "output_model", "Output for trained perceptron"
    " model.", "M");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // First, get all parameters and validate them.
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");

  // Check parameter validity.
  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "maximum number of iterations must be nonnegative");

  // Now set up our model.
  PerceptronModel* p = new PerceptronModel();

  if (params.Has("labels"))
  {
    Log::Info << "' with labels in '"
        << params.GetPrintable<Row<size_t>>("labels") << "'";
  }
  else
  {
    Log::Info << "'";
  }
  Log::Info << " for a maximum of " << maxIterations << " iterations."
      << endl;

  mat trainingData = std::move(params.Get<mat>("training"));

  // Load labels.
  Row<size_t> labelsIn;

  // Did the user pass in labels?
  if (params.Has("labels"))
  {
    labelsIn = std::move(params.Get<Row<size_t>>("labels"));

    // Checking the size of the responses and training data.
    if (labelsIn.n_cols != trainingData.n_cols)
    {
      // Clean memory.
      delete p;
      Log::Fatal << "The responses must have the same number of columns "
          "as the training set." << endl;
    }
  }
  else
  {
    // Checking the size of training data if no labels are passed.
    if (trainingData.n_rows < 2)
    {
      // Clean memory.
      delete p;
      Log::Fatal << "Can't get responses from training data "
          "since it has less than 2 rows." << endl;
    }

    // Use the last row of the training data as the labels.
    Log::Info << "Using the last dimension of training set as labels."
        << endl;
    labelsIn = ConvTo<Row<size_t>>::From(
        trainingData.row(trainingData.n_rows - 1));
    trainingData.shed_row(trainingData.n_rows - 1);
  }

  // Normalize the labels.
  Row<size_t> labels;
  NormalizeLabels(labelsIn, labels, p->Map());
  const size_t numClasses = p->Map().n_elem;

  // Create and train the classifier.
  timers.Start("training");
  p->P() = Perceptron<>(trainingData, labels, numClasses, maxIterations);
  timers.Stop("training");

  // Lastly, save the output model.
  params.Get<PerceptronModel*>("output_model") = p;
}
