/**
 * @file methods/perceptron/perceptron_main.cpp
 * @author Udit Saxena
 *
 * This program runs the Simple Perceptron Classifier.
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
#define BINDING_NAME perceptron

#include <mlpack/core/util/mlpack_main.hpp>

#include "perceptron.hpp"

using namespace mlpack;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_USER_NAME("Perceptron");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of a perceptron---a single level neural network--=for "
    "classification.  Given labeled data, a perceptron can be trained and saved"
    " for future use; or, a pre-trained perceptron can be used for "
    "classification on new points.");

// Long description.
BINDING_LONG_DESC(
    "This program implements a perceptron, which is a single level neural "
    "network. The perceptron makes its predictions based on a linear predictor "
    "function combining a set of weights with the feature vector.  The "
    "perceptron learning rule is able to converge, given enough iterations "
    "(specified using the " + PRINT_PARAM_STRING("max_iterations") +
    " parameter), if the data supplied is linearly separable.  The perceptron "
    "is parameterized by a matrix of weight vectors that denote the numerical "
    "weights of the neural network."
    "\n\n"
    "This program allows loading a perceptron from a model (via the " +
    PRINT_PARAM_STRING("input_model") + " parameter) or training a perceptron "
    "given training data (via the " + PRINT_PARAM_STRING("training") +
    " parameter), or both those things at once.  In addition, this program "
    "allows classification on a test dataset (via the " +
    PRINT_PARAM_STRING("test") + " parameter) and the classification results "
    "on the test set may be saved with the " +
    PRINT_PARAM_STRING("predictions") +
    " output parameter.  The perceptron model may be saved with the " +
    PRINT_PARAM_STRING("output_model") + " output parameter.");

// Example.
BINDING_EXAMPLE(
    "The training data given with the " + PRINT_PARAM_STRING("training") +
    " option may have class labels as its last dimension (so, if the training "
    "data is in CSV format, labels should be the last column).  Alternately, "
    "the " + PRINT_PARAM_STRING("labels") + " parameter may be used to specify "
    "a separate matrix of labels."
    "\n\n"
    "All these options make it easy to train a perceptron, and then re-use that"
    " perceptron for later classification.  The invocation below trains a "
    "perceptron on " + PRINT_DATASET("training_data") + " with labels " +
    PRINT_DATASET("training_labels") + ", and saves the model to " +
    PRINT_MODEL("perceptron_model") + "."
    "\n\n" +
    PRINT_CALL("perceptron", "training", "training_data", "labels",
        "training_labels", "output_model", "perceptron_model") +
    "\n\n"
    "Then, this model can be re-used for classification on the test data " +
    PRINT_DATASET("test_data") + ".  The example below does precisely that, "
    "saving the predicted classes to " + PRINT_DATASET("predictions") + "."
    "\n\n" +
    PRINT_CALL("perceptron", "input_model", "perceptron_model", "test",
        "test_data", "predictions", "predictions") +
    "\n\n"
    "Note that all of the options may be specified at once: predictions may be "
    "calculated right after training a model, and model training can occur even"
    " if an existing perceptron model is passed with the " +
    PRINT_PARAM_STRING("input_model") + " parameter.  However, note that the "
    "number of classes and the dimensionality of all data must match.  So you "
    "cannot pass a perceptron model trained on 2 classes and then re-train with"
    " a 4-class dataset.  Similarly, attempting classification on a "
    "3-dimensional dataset with a perceptron that has been trained on 8 "
    "dimensions will cause an error.");

// See also...
BINDING_SEE_ALSO("@adaboost", "#adaboost");
BINDING_SEE_ALSO("Perceptron on Wikipedia",
    "https://en.wikipedia.org/wiki/Perceptron");
BINDING_SEE_ALSO("Perceptron C++ class documentation",
    "@doc/user/methods/perceptron.md");

// When we save a model, we must also save the class mappings.  So we use this
// auxiliary structure to store both the perceptron and the mapping, and we'll
// save this.
class PerceptronModel
{
 private:
  Perceptron<> p;
  Col<size_t> map;

 public:
  Perceptron<>& P() { return p; }
  const Perceptron<>& P() const { return p; }

  Col<size_t>& Map() { return map; }
  const Col<size_t>& Map() const { return map; }

  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(p));
    ar(CEREAL_NVP(map));
  }
};

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set.", "t");
PARAM_UROW_IN("labels", "A matrix containing labels for the training set.",
    "l");
PARAM_INT_IN("max_iterations", "The maximum number of iterations the "
    "perceptron is to be run", "n", 1000);

// Model loading/saving.
PARAM_MODEL_IN(PerceptronModel, "input_model", "Input perceptron model.", "m");
PARAM_MODEL_OUT(PerceptronModel, "output_model", "Output for trained perceptron"
    " model.", "M");

// Testing/classification parameters.
PARAM_MATRIX_IN("test", "A matrix containing the test set.", "T");
PARAM_UROW_OUT("predictions", "The matrix in which the predicted labels for the"
    " test set will be written.", "P");

void BINDING_FUNCTION(util::Params& params, util::Timers& timers)
{
  // First, get all parameters and validate them.
  const size_t maxIterations = (size_t) params.Get<int>("max_iterations");

  // We must either load a model or train a model.
  RequireAtLeastOnePassed(params, { "input_model", "training" }, true);

  // If the user isn't going to save the output model or any predictions, we
  // should issue a warning.
  RequireAtLeastOnePassed(params, { "output_model", "predictions" }, false,
      "no output will be saved");
  ReportIgnoredParam(params, {{ "test", false }}, "predictions");

  // Check parameter validity.
  RequireParamValue<int>(params, "max_iterations", [](int x) { return x >= 0; },
      true, "maximum number of iterations must be nonnegative");

  // Now, load our model, if there is one.
  PerceptronModel* p;
  if (params.Has("input_model"))
  {
    Log::Info << "Using saved perceptron from "
        << params.GetPrintable<PerceptronModel*>("input_model") << "."
        << endl;

    p = params.Get<PerceptronModel*>("input_model");
  }
  else
  {
    p = new PerceptronModel();
  }

  // Next, load the training data and labels (if they have been given).
  if (params.Has("training"))
  {
    // Get and cache the value of GetPrintableParam<mat>("training").
    std::ostringstream oss;
    oss << params.GetPrintable<mat>("training");
    std::string trainingOutput = oss.str();

    Log::Info << "Training perceptron on dataset '" << trainingOutput;
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
        // Clean memory if needed.
        if (!params.Has("input_model"))
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
        // Clean memory if needed.
        if (!params.Has("input_model"))
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
    data::NormalizeLabels(labelsIn, labels, p->Map());
    const size_t numClasses = p->Map().n_elem;

    // Now, if we haven't already created a perceptron, do it.  Otherwise, make
    // sure the dimensions are right, then continue training.
    if (!params.Has("input_model"))
    {
      // Create and train the classifier.
      timers.Start("training");
      p->P() = Perceptron<>(trainingData, labels, numClasses, maxIterations);
      timers.Stop("training");
    }
    else
    {
      // Check dimensionality.
      if (p->P().Weights().n_rows != trainingData.n_rows)
      {
        Log::Fatal << "Perceptron from '"
            << params.GetPrintable<PerceptronModel*>("input_model")
            << "' is built on data with " << p->P().Weights().n_rows
            << " dimensions, but data in '" << trainingOutput << "' has "
            << trainingData.n_rows << "dimensions!" << endl;
      }

      // Check the number of labels.
      if (numClasses > p->P().Weights().n_cols)
      {
        Log::Fatal << "Perceptron from '"
            << params.GetPrintable<PerceptronModel*>("input_model") << "' "
            << "has " << p->P().Weights().n_cols << " classes, but the training"
            << " data has " << numClasses + 1 << " classes!" << endl;
      }

      // Now train.
      timers.Start("training");
      p->P().MaxIterations() = maxIterations;
      p->P().Train(trainingData, labels.t(), numClasses);
      timers.Stop("training");
    }
  }

  // Now, the training procedure is complete.  Do we have any test data?
  if (params.Has("test"))
  {
    mat& testData = params.Get<arma::mat>("test");
    Log::Info << "Classifying dataset '"
        << params.GetPrintable<arma::mat>("test") << "'." << endl;

    if (testData.n_rows != p->P().Weights().n_rows)
    {
      // Clean memory if needed.
      const size_t perceptronDimensionality = p->P().Weights().n_rows;
      if (!params.Has("input_model"))
        delete p;

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
    data::RevertLabels(predictedLabels, p->Map(), results);

    // Save the predicted labels.
    if (params.Has("predictions"))
      params.Get<arma::Row<size_t>>("predictions") = std::move(results);
  }

  // Lastly, save the output model.
  params.Get<PerceptronModel*>("output_model") = p;
}
