/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file nbc_main.cpp
 *
 * This program runs the Simple Naive Bayes Classifier.
 *
 * This classifier does parametric naive bayes classification assuming that the
 * features are sampled from a Gaussian distribution.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
PARAM_STRING("input_model_file", "File containing input Naive Bayes model.",
    "m", "");
PARAM_STRING("output_model_file", "File to save trained Naive Bayes model to.",
    "M", "");

// Training parameters.
PARAM_STRING("training_file", "A file containing the training set.", "t", "");
PARAM_STRING("labels_file", "A file containing labels for the training set.",
    "l", "");
PARAM_FLAG("incremental_variance", "The variance of each class will be "
    "calculated incrementally.", "I");

// Test parameters.
PARAM_STRING("test_file", "A file containing the test set.", "T", "");
PARAM_STRING("output_file", "The file in which the predicted labels for the "
    "test set will be written.", "o", "");

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
  if (CLI::HasParam("training_file") && CLI::HasParam("input_model_file"))
    Log::Fatal << "Cannot specify both --training_file (-t) and "
        << "--input_model_file (-m)!" << endl;

  if (!CLI::HasParam("training_file") && !CLI::HasParam("input_model_file"))
    Log::Fatal << "Neither --training_file (-t) nor --input_model_file (-m) are"
        << " specified!" << endl;

  if (!CLI::HasParam("training_file") && CLI::HasParam("labels_file"))
    Log::Warn << "--labels_file (-l) ignored because --training_file (-t) is "
        << "not specified." << endl;
  if (!CLI::HasParam("training_file") && CLI::HasParam("incremental_variance"))
    Log::Warn << "--incremental_variance (-I) ignored because --training_file "
        << "(-t) is not specified." << endl;

  if (!CLI::HasParam("output_file") && !CLI::HasParam("output_model_file"))
    Log::Warn << "Neither --output_file (-o) nor --output_model_file (-M) "
        << "specified; no output will be saved!" << endl;

  if (CLI::HasParam("output_file") && !CLI::HasParam("test_file"))
    Log::Warn << "--output_file (-o) ignored because no test file specified "
        << "with --test_file (-T)." << endl;

  if (!CLI::HasParam("output_file") && CLI::HasParam("test_file"))
    Log::Warn << "--test_file (-T) specified, but classification results will "
        << "not be saved because --output_file (-o) is not specified." << endl;

  // Either we have to train a model, or load a model.
  NBCModel model;
  if (CLI::HasParam("training_file"))
  {
    const string trainingFile = CLI::GetParam<string>("training_file");
    mat trainingData;
    data::Load(trainingFile, trainingData, true);

    Row<size_t> labels;

    // Did the user pass in labels?
    const string labelsFilename = CLI::GetParam<string>("labels_file");
    if (labelsFilename != "")
    {
      // Load labels.
      mat rawLabels;
      data::Load(labelsFilename, rawLabels, true, false);

      // Do the labels need to be transposed?
      if (rawLabels.n_cols == 1)
        rawLabels = rawLabels.t();

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
  if (CLI::HasParam("test_file"))
  {
    const string testingDataFilename = CLI::GetParam<std::string>("test_file");
    mat testingData;
    data::Load(testingDataFilename, testingData, true);

    if (testingData.n_rows != model.nbc.Means().n_rows)
      Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
          << "must be the same as training data (" << model.nbc.Means().n_rows
          << ")!" << std::endl;

    // Time the running of the Naive Bayes Classifier.
    Row<size_t> results;
    Timer::Start("nbc_testing");
    model.nbc.Classify(testingData, results);
    Timer::Stop("nbc_testing");

    if (CLI::HasParam("output_file"))
    {
      // Un-normalize labels to prepare output.
      Row<size_t> rawResults;
      data::RevertLabels(results, model.mappings, rawResults);

      // Output results.
      const string outputFilename = CLI::GetParam<string>("output_file");
      data::Save(outputFilename, rawResults, true);
    }
  }

  if (CLI::HasParam("output_model_file"))
    data::Save(CLI::GetParam<string>("output_model_file"), "nbc_model", model,
        false);
}
