/**
 * @file decision_stump_main.cpp
 * @author Udit Saxena
 *
 * Main executable for the decision stump.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include "decision_stump.hpp"

using namespace mlpack;
using namespace mlpack::decision_stump;
using namespace std;
using namespace arma;

PROGRAM_INFO("Decision Stump",
    "This program implements a decision stump, which is a single-level decision"
    " tree.  The decision stump will split on one dimension of the input data, "
    "and will split into multiple buckets.  The dimension and bins are selected"
    " by maximizing the information gain of the split.  Optionally, the minimum"
    " number of training points in each bin can be specified with the "
    "--bucket_size (-b) parameter.\n"
    "\n"
    "The decision stump is parameterized by a splitting dimension and a vector "
    "of values that denote the splitting values of each bin.\n"
    "\n"
    "This program enables several applications: a decision tree may be trained "
    "or loaded, and then that decision tree may be used to classify a given set"
    " of test points.  The decision tree may also be saved to a file for later "
    "usage.\n"
    "\n"
    "To train a decision stump, training data should be passed with the "
    "--training_file (-t) option, and their corresponding labels should be "
    "passed with the --labels_file (-l) option.  Optionally, if --labels_file "
    "is not specified, the labels are assumed to be the last dimension of the "
    "training dataset.  The --bucket_size (-b) parameter controls the minimum "
    "number of training points in each decision stump bucket.\n"
    "\n"
    "For classifying a test set, a decision stump may be loaded with the "
    "--input_model_file (-m) parameter (useful for the situation where a "
    "stump has not just been trained), and a test set may be specified with the"
    " --test_file (-T) parameter.  The predicted labels will be saved to the "
    "file specified with the --predictions_file (-p) parameter.\n"
    "\n"
    "Because decision stumps are trained in batch, retraining does not make "
    "sense and thus it is not possible to pass both --training_file and "
    "--input_model_file; instead, simply build a new decision stump with the "
    "training data.\n"
    "\n"
    "A trained decision stump can be saved with the --output_model_file (-M) "
    "option.  That stump may later be re-used in subsequent calls to this "
    "program (or others).");

// Datasets we might load.
PARAM_MATRIX_IN("training", "The dataset to train on.", "t");
PARAM_UMATRIX_IN("labels", "Labels for the training set. If not specified, the "
    "labels are assumed to be the last row of the training data.", "l");
PARAM_MATRIX_IN("test", "A dataset to calculate predictions for.", "T");

// Output.
PARAM_UMATRIX_OUT("predictions", "The output matrix that will hold the "
    "predicted labels for the test set.", "p");

// We may load or save a model.
PARAM_STRING_IN("input_model_file", "File containing decision stump model to "
    "load.", "m", "");
PARAM_STRING_OUT("output_model_file", "File to save trained decision stump "
    "model to.", "M");

PARAM_INT_IN("bucket_size", "The minimum number of training points in each "
    "decision stump bucket.", "b", 6);

/**
 * This is the structure that actually saves to disk.  We have to save the
 * label mappings, too, otherwise everything we load at test time in a future
 * run will end up being borked.
 */
struct DSModel
{
  //! The mappings.
  arma::Col<size_t> mappings;
  //! The stump.
  DecisionStump<> stump;

  //! Serialize the model.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */)
  {
    ar & data::CreateNVP(mappings, "mappings");
    ar & data::CreateNVP(stump, "stump");
  }
};

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Check that the parameters are reasonable.
  if (CLI::HasParam("training") && CLI::HasParam("input_model_file"))
  {
    Log::Fatal << "Both --training_file and --input_model_file are specified, "
        << "but a trained model cannot be retrained.  Only one of these options"
        << " may be specified." << endl;
  }

  if (!CLI::HasParam("training") && !CLI::HasParam("input_model_file"))
  {
    Log::Fatal << "Neither --training_file nor --input_model_file are given; "
        << "one must be specified." << endl;
  }

  if (!CLI::HasParam("output_model_file") && !CLI::HasParam("predictions"))
  {
    Log::Warn << "Neither --output_model_file nor --predictions_file are "
        << "specified; no results will be saved!" << endl;
  }

  // We must either load a model, or train a new stump.
  DSModel model;
  if (CLI::HasParam("training"))
  {
    mat trainingData = std::move(CLI::GetParam<mat>("training"));

    // Load labels, if necessary.
    Mat<size_t> labelsIn;
    if (CLI::HasParam("labels"))
    {
      labelsIn = std::move(CLI::GetParam<Mat<size_t>>("labels"));

      // Do the labels need to be transposed?
      if (labelsIn.n_cols == 1)
        labelsIn = labelsIn.t();
      if (labelsIn.n_cols == 1)
        Log::Fatal << "Labels must be one-dimensional!" << endl;
    }
    else
    {
      // Extract the labels as the last
      Log::Info << "Using the last dimension of training set as labels."
          << endl;

      labelsIn = arma::conv_to<arma::Mat<size_t>>::from(
          trainingData.row(trainingData.n_rows - 1).t());
      trainingData.shed_row(trainingData.n_rows - 1);
    }

    // Normalize the labels.
    Row<size_t> labels;
    data::NormalizeLabels(labelsIn.row(0), labels, model.mappings);

    const size_t bucketSize = CLI::GetParam<int>("bucket_size");
    const size_t classes = labels.max() + 1;

    Timer::Start("training");
    model.stump.Train(trainingData, labels, classes, bucketSize);
    Timer::Stop("training");
  }
  else
  {
    const string inputModelFile = CLI::GetParam<string>("input_model_file");
    data::Load(inputModelFile, "decision_stump_model", model, true);
  }

  // Now, do we need to do any testing?
  if (CLI::HasParam("test"))
  {
    // Load the test file.
    mat testingData = std::move(CLI::GetParam<arma::mat>("test"));

    if (testingData.n_rows <= model.stump.SplitDimension())
      Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
          << "is too low; the trained stump requires at least "
          << model.stump.SplitDimension() << " dimensions!" << endl;

    Row<size_t> predictedLabels(testingData.n_cols);
    Timer::Start("testing");
    model.stump.Classify(testingData, predictedLabels);
    Timer::Stop("testing");

    // Denormalize predicted labels, if we want to save them.
    if (CLI::HasParam("predictions"))
    {
      Row<size_t> actualLabels;
      data::RevertLabels(predictedLabels, model.mappings, actualLabels);

      // Save the predicted labels in a transposed form as output.
      CLI::GetParam<Mat<size_t>>("predictions") = std::move(actualLabels);
    }
  }

  // Save the model, if desired.
  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"),
        "decision_stump_model", model);
}
