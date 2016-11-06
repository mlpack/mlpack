/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file nbc_main.cpp
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
#include <mlpack/core.hpp>

#include "naive_bayes_classifier.hpp"

PROGRAM_INFO("Parametric Naive Bayes Classifier",
    "This program trains the Naive Bayes classifier on the given labeled "
    "training set, or loads a model from the given model file, and then may use"
    " that trained model to classify the points in a given test set."
    "\n\n"
    "Labels are expected to be the last row of the training set (--train_file),"
    " but labels can also be passed in separately as their own file "
    "(--labels_file).  If training is not desired, a pre-existing model can be "
    "loaded with the --input_model_file (-m) option."
    "\n\n"
    "The '--incremental_variance' option can be used to force the training to "
    "use an incremental algorithm for calculating variance.  This is slower, "
    "but can help avoid loss of precision in some cases."
    "\n\n"
    "If classifying a test set is desired, the test set should be in the file "
    "specified with the --test_file (-T) option, and the classifications will "
    "be saved to the file specified with the --output_file (-o) option.  If "
    "saving a trained model is desired, the --output_model_file (-M) option "
    "should be given.");

// Model loading/saving.
PARAM_STRING_IN("input_model_file", "File containing input Naive Bayes model.",
    "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained Naive Bayes model "
    "to.", "M");

// Training parameters.
PARAM_MATRIX_IN("training", "A matrix containing the training set.", "t");
PARAM_UMATRIX_IN("labels", "A file containing labels for the training set.",
    "l");
PARAM_FLAG("incremental_variance", "The variance of each class will be "
    "calculated incrementally.", "I");

// Test parameters.
PARAM_MATRIX_IN("test", "A matrix containing the test set.", "T");
PARAM_UMATRIX_OUT("output", "The matrix in which the predicted labels for the"
    " test set will be written.", "o");

using namespace mlpack;
using namespace mlpack::naive_bayes;
using namespace std;
using namespace arma;

// A struct for saving the model with mappings.
struct NBCModel
{
  //! The model itself.
  NaiveBayesClassifier<> nbc;
  //! The mappings for labels.
  Col<size_t> mappings;

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(nbc, "nbc");
    ar & data::CreateNVP(mappings, "mappings");
  }
};

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Check input parameters.
  if (CLI::HasParam("training") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Cannot specify both --training_file (-t) and "
        << "--input_model_file (-m)!" << endl;

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Neither --training_file (-t) nor --input_model_file (-m) are"
        << " specified!" << endl;

  if (!CLI::HasParam("training") && CLI::HasParam("labels"))
    Log::Warn << "--labels_file (-l) ignored because --training_file (-t) is "
        << "not specified." << endl;
  if (!CLI::HasParam("training") && CLI::HasParam("incremental_variance"))
    Log::Warn << "--incremental_variance (-I) ignored because --training_file "
        << "(-t) is not specified." << endl;

  if (!CLI::HasParam("output") && !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither --output_file (-o) nor --output_model_file (-M) "
        << "specified; no output will be saved!" << endl;

  if (CLI::HasParam("output") && !CLI::HasParam("test"))
    Log::Warn << "--output_file (-o) ignored because no test file specified "
        << "with --test_file (-T)." << endl;

  if (!CLI::HasParam("output") && CLI::HasParam("test"))
    Log::Warn << "--test_file (-T) specified, but classification results will "
        << "not be saved because --output_file (-o) is not specified." << endl;

  // Either we have to train a model, or load a model.
  NBCModel model;
  if (CLI::HasParam("training"))
  {
    mat trainingData = std::move(CLI::GetParam<mat>("training"));

    Row<size_t> labels;

    // Did the user pass in labels?
    if (CLI::HasParam("labels"))
    {
      // Load labels.
      Mat<size_t> rawLabels = std::move(CLI::GetParam<Mat<size_t>>("labels"));

      // Do the labels need to be transposed?
      if (rawLabels.n_cols == 1)
        rawLabels = rawLabels.t();
      if (rawLabels.n_rows > 1)
        Log::Fatal << "Labels must be one-dimensional!" << endl;

      data::NormalizeLabels(rawLabels.row(0), labels, model.mappings);
    }
    else
    {
      // Use the last row of the training data as the labels.
      Log::Info << "Using last dimension of training data as training labels."
          << endl;
      data::NormalizeLabels(trainingData.row(trainingData.n_rows - 1), labels,
          model.mappings);
      // Remove the label row.
      trainingData.shed_row(trainingData.n_rows - 1);
    }

    const bool incrementalVariance = CLI::HasParam("incremental_variance");

    Timer::Start("nbc_training");
    model.nbc = NaiveBayesClassifier<>(trainingData, labels,
        model.mappings.n_elem, incrementalVariance);
    Timer::Stop("nbc_training");
  }
  else
  {
    // Load the model from file.
    data::Load(CLI::GetParam<string>("input_model_file"), "nbc_model", model);
  }

  // Do we need to do testing?
  if (CLI::HasParam("test"))
  {
    mat testingData = std::move(CLI::GetParam<mat>("test"));

    if (testingData.n_rows != model.nbc.Means().n_rows)
      Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
          << "must be the same as training data (" << model.nbc.Means().n_rows
          << ")!" << std::endl;

    // Time the running of the Naive Bayes Classifier.
    Row<size_t> results;
    Timer::Start("nbc_testing");
    model.nbc.Classify(testingData, results);
    Timer::Stop("nbc_testing");

    if (CLI::HasParam("output"))
    {
      // Un-normalize labels to prepare output.
      Row<size_t> rawResults;
      data::RevertLabels(results, model.mappings, rawResults);

      // Output results.
      CLI::GetParam<Mat<size_t>>("output") = std::move(rawResults);
    }
  }

  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "nbc_model", model,
        false);
}
