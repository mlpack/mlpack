/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file nbc_main.cpp
 *
 * This program runs the Simple Naive Bayes Classifier.
 *
 * This classifier does parametric naive bayes classification assuming that the
 * features are sampled from a Gaussian distribution.
 */
#include <mlpack/core.hpp>

#include "naive_bayes_classifier.hpp"

PROGRAM_INFO("Parametric Naive Bayes Classifier",
    "This program trains the Naive Bayes classifier on the given labeled "
    "training set and then uses the trained classifier to classify the points "
    "in the given test set."
    "\n\n"
    "Labels are expected to be the last row of the training set (--train_file),"
    " but labels can also be passed in separately as their own file "
    "(--labels_file)."
    "\n\n"
    "The '--incremental_variance' option can be used to force the training to "
    "use an incremental algorithm for calculating variance.  This is slower, "
    "but can help avoid loss of precision in some cases.");

PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");

PARAM_STRING("labels_file", "A file containing labels for the training set.",
    "l", "");
PARAM_STRING("output_file", "The file in which the predicted labels for the "
    "test set will be written.", "o", "output.csv");
PARAM_FLAG("incremental_variance", "The variance of each class will be "
    "calculated incrementally.", "I");

using namespace mlpack;
using namespace mlpack::naive_bayes;
using namespace std;
using namespace arma;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  // Check input parameters.
  const string trainingDataFilename = CLI::GetParam<string>("train_file");
  mat trainingData;
  data::Load(trainingDataFilename, trainingData, true);

  // Normalize labels.
  Row<size_t> labels;
  vec mappings;

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

    data::NormalizeLabels(rawLabels.row(0), labels, mappings);
  }
  else
  {
    // Use the last row of the training data as the labels.
    Log::Info << "Using last dimension of training data as training labels."
        << endl;
    data::NormalizeLabels(trainingData.row(trainingData.n_rows - 1), labels,
        mappings);
    // Remove the label row.
    trainingData.shed_row(trainingData.n_rows - 1);
  }

  const string testingDataFilename = CLI::GetParam<std::string>("test_file");
  mat testingData;
  data::Load(testingDataFilename, testingData, true);

  if (testingData.n_rows != trainingData.n_rows)
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << trainingData.n_rows
        << ")!" << std::endl;

  const bool incrementalVariance = CLI::HasParam("incremental_variance");

  // Create and train the classifier.
  Timer::Start("training");
  NaiveBayesClassifier<> nbc(trainingData, labels, mappings.n_elem,
      incrementalVariance);
  Timer::Stop("training");

  // Time the running of the Naive Bayes Classifier.
  Row<size_t> results;
  Timer::Start("testing");
  nbc.Classify(testingData, results);
  Timer::Stop("testing");

  // Un-normalize labels to prepare output.
  rowvec rawResults;
  data::RevertLabels(results, mappings, rawResults);

  // Output results.  Transpose: one result per line.
  const string outputFilename = CLI::GetParam<string>("output_file");
  data::Save(outputFilename, rawResults, true, false);
}
