/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file methods/naive_bayes/nbc_main.cpp
 *
 * This program runs the Simple Naive Bayes Classifier.
 *
 * This classifier does parametric naive bayes classification assuming that the
 * features are sampled from a Gaussian distribution.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/core/data/normalize_labels.hpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "naive_bayes_classifier.hpp"

using namespace mlpack;
using namespace mlpack::naive_bayes;
using namespace mlpack::util;
using namespace std;
using namespace arma;

// Program Name.
BINDING_NAME("Parametric Naive Bayes Classifier");

// Short description.
BINDING_SHORT_DESC(
    "An implementation of the Naive Bayes Classifier, used for classification. "
    "Given labeled data, an NBC model can be trained and saved, or, a "
    "pre-trained model can be used for classification.");

// Long description.
BINDING_LONG_DESC(
    "This program trains the Naive Bayes classifier on the given labeled "
    "training set, or loads a model from the given model file, and then may use"
    " that trained model to classify the points in a given test set."
    "\n\n"
    "The training set is specified with the " +
    PRINT_PARAM_STRING("training") + " parameter.  Labels may be either the "
    "last row of the training set, or alternately the " +
    PRINT_PARAM_STRING("labels") + " parameter may be specified to pass a "
    "separate matrix of labels."
    "\n\n"
    "If training is not desired, a pre-existing model may be loaded with the " +
    PRINT_PARAM_STRING("input_model") + " parameter."
    "\n\n"
    "\n\n"
    "The " + PRINT_PARAM_STRING("incremental_variance") + " parameter can be "
    "used to force the training to use an incremental algorithm for calculating"
    " variance.  This is slower, but can help avoid loss of precision in some "
    "cases."
    "\n\n"
    "If classifying a test set is desired, the test set may be specified with "
    "the " + PRINT_PARAM_STRING("test") + " parameter, and the classifications"
    " may be saved with the " + PRINT_PARAM_STRING("predictions") +"predictions"
    "  parameter.  If saving the trained model is desired, this may be "
    "done with the " + PRINT_PARAM_STRING("output_model") + " output "
    "parameter."
    "\n\n"
    "Note: the " + PRINT_PARAM_STRING("output") + " and " +
    PRINT_PARAM_STRING("output_probs") + " parameters are deprecated and will "
    "be removed in mlpack 4.0.0.  Use " + PRINT_PARAM_STRING("predictions") +
    " and " + PRINT_PARAM_STRING("probabilities") + " instead.");

// Example.
BINDING_EXAMPLE(
    "For example, to train a Naive Bayes classifier on the dataset " +
    PRINT_DATASET("data") + " with labels " + PRINT_DATASET("labels") + " "
    "and save the model to " + PRINT_MODEL("nbc_model") + ", the following "
    "command may be used:"
    "\n\n" +
    PRINT_CALL("nbc", "training", "data", "labels", "labels", "output_model",
        "nbc_model") +
    "\n\n"
    "Then, to use " + PRINT_MODEL("nbc_model") + " to predict the classes of "
    "the dataset " + PRINT_DATASET("test_set") + " and save the predicted "
    "classes to " + PRINT_DATASET("predictions") + ", the following command "
    "may be used:"
    "\n\n" +
    PRINT_CALL("nbc", "input_model", "nbc_model", "test", "test_set", "output",
        "predictions"));

// See also...
BINDING_SEE_ALSO("@softmax_regression", "#softmax_regression");
BINDING_SEE_ALSO("@random_forest", "#random_forest");
BINDING_SEE_ALSO("Naive Bayes classifier on Wikipedia",
        "https://en.wikipedia.org/wiki/Naive_Bayes_classifier");
BINDING_SEE_ALSO("mlpack::naive_bayes::NaiveBayesClassifier C++ class "
        "documentation", "@doxygen/classmlpack_1_1naive__bayes_1_1"
        "NaiveBayesClassifier.html");

// A struct for saving the model with mappings.
struct NBCModel
{
  //! The model itself.
  NaiveBayesClassifier<> nbc;
  //! The mappings for labels.
  Col<size_t> mappings;

  //! Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & BOOST_SERIALIZATION_NVP(nbc);
    ar & BOOST_SERIALIZATION_NVP(mappings);
  }
};

// Model loading/saving.
PARAM_MODEL_IN(NBCModel, "input_model", "Input Naive Bayes "
    "model.", "m");
PARAM_MODEL_OUT(NBCModel, "output_model", "File to save trained "
    "Naive Bayes model to.", "M");

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set.", "t");
PARAM_UROW_IN("labels", "A file containing labels for the training set.",
    "l");
PARAM_FLAG("incremental_variance", "The variance of each class will be "
    "calculated incrementally.", "I");

// Test parameters.
PARAM_MATRIX_IN("test", "A matrix containing the test set.", "T");
// The parameter 'output' is deprecated and will be removed in mlpack 4.
PARAM_UROW_OUT("output", "The matrix in which the predicted labels for the"
    " test set will be written (deprecated).", "o");
PARAM_UROW_OUT("predictions", "The matrix in which the predicted labels for the"
    " test set will be written.", "a");
// The parameter 'output_probs' is deprecated and can be removed in mlpack 4.
PARAM_MATRIX_OUT("output_probs", "The matrix in which the predicted probability"
    " of labels for the test set will be written (deprecated).", "");
PARAM_MATRIX_OUT("probabilities", "The matrix in which the predicted"
    " probability of labels for the test set will be written.", "p");

static void mlpackMain()
{
  // Check input parameters.
  RequireOnlyOnePassed({ "training", "input_model" }, true);
  ReportIgnoredParam({{ "training", false }}, "labels");
  ReportIgnoredParam({{ "training", false }}, "incremental_variance");
  RequireAtLeastOnePassed({ "output", "predictions", "output_model",
      "output_probs", "probabilities" }, false, "no output will be saved");
  ReportIgnoredParam({{ "test", false }}, "output");
  ReportIgnoredParam({{ "test", false }}, "predictions");
  if (IO::HasParam("input_model") && !IO::HasParam("test"))
    Log::Warn << "No test set given; no task will be performed!" << std::endl;

  // Either we have to train a model, or load a model.
  NBCModel* model;
  if (IO::HasParam("training"))
  {
    model = new NBCModel();
    mat trainingData = std::move(IO::GetParam<mat>("training"));

    Row<size_t> labels;

    // Did the user pass in labels?
    if (IO::HasParam("labels"))
    {
      // Load labels.
      Row<size_t> rawLabels = std::move(IO::GetParam<Row<size_t>>("labels"));
      data::NormalizeLabels(rawLabels, labels, model->mappings);
    }
    else
    {
      // Use the last row of the training data as the labels.
      Log::Info << "Using last dimension of training data as training labels."
          << endl;
      data::NormalizeLabels(trainingData.row(trainingData.n_rows - 1), labels,
          model->mappings);
      // Remove the label row.
      trainingData.shed_row(trainingData.n_rows - 1);
    }
    const bool incrementalVariance = IO::HasParam("incremental_variance");

    Timer::Start("nbc_training");
    model->nbc = NaiveBayesClassifier<>(trainingData, labels,
        model->mappings.n_elem, incrementalVariance);
    Timer::Stop("nbc_training");
  }
  else
  {
    // Load the model from file.
    model = IO::GetParam<NBCModel*>("input_model");
  }

  // Do we need to do testing?
  if (IO::HasParam("test"))
  {
    mat testingData = std::move(IO::GetParam<mat>("test"));

    if (testingData.n_rows != model->nbc.Means().n_rows)
    {
      Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
          << "must be the same as training data (" << model->nbc.Means().n_rows
          << ")!" << std::endl;
    }

    // Time the running of the Naive Bayes Classifier.
    Row<size_t> predictions;
    mat probabilities;
    Timer::Start("nbc_testing");
    model->nbc.Classify(testingData, predictions, probabilities);
    Timer::Stop("nbc_testing");

    if (IO::HasParam("output") || IO::HasParam("predictions"))
    {
      // Un-normalize labels to prepare output.
      Row<size_t> rawResults;
      data::RevertLabels(predictions, model->mappings, rawResults);

      if (IO::HasParam("predictions"))
        IO::GetParam<Row<size_t>>("predictions") = rawResults;
      if (IO::HasParam("output"))
        IO::GetParam<Row<size_t>>("output") = std::move(rawResults);
    }
    if (IO::HasParam("output_probs") || IO::HasParam("probabilities"))
    {
      if (IO::HasParam("probabilities"))
        IO::GetParam<mat>("probabilities") = probabilities;
      if (IO::HasParam("output_probs"))
        IO::GetParam<mat>("output_probs") = std::move(probabilities);
    }
  }

  IO::GetParam<NBCModel*>("output_model") = model;
}
