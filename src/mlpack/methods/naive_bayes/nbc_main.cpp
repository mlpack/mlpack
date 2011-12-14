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
    "in the given test set.\n"
    "\n"
    "Labels are expected to be the last row of the training set (--train_file),"
    " but labels can also be passed in separately as their own file "
    "(--labels_file).");

PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");

PARAM_STRING("labels_file", "A file containing labels for the training set.",
    "l", "");
PARAM_STRING("output", "The file in which the output of the test would "
    "be written, defaults to 'output.csv')", "o", "output.csv");

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
  data::Load(trainingDataFilename.c_str(), trainingData, true);

  // Did the user pass in labels?
  const string labelsFilename = CLI::GetParam<string>("labels_file");
  if (labelsFilename != "")
  {
    // Load labels.
    arma::mat labels;
    data::Load(labelsFilename.c_str(), labels, true);

    // Not incredibly efficient...
    if (labels.n_rows == 1)
      trainingData.insert_rows(trainingData.n_rows, trans(labels));
    else if (labels.n_cols == 1)
      trainingData.insert_rows(trainingData.n_rows, labels);
    else
      Log::Fatal << "Labels must have only one column or row!" << endl;
  }

  const string testingDataFilename = CLI::GetParam<std::string>("test_file");
  mat testingData;
  data::Load(testingDataFilename.c_str(), testingData, true);

  if (testingData.n_rows != trainingData.n_rows - 1)
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << trainingData.n_rows - 1
        << ")!" << std::endl;

  // Calculate number of classes.
  size_t classes = (size_t) max(trainingData.row(trainingData.n_rows - 1)) + 1;

  // Create and train the classifier.
  Timer::Start("training");
  NaiveBayesClassifier<> nbc(trainingData, classes);
  Timer::Stop("training");

  // Timing the running of the Naive Bayes Classifier.
  arma::Col<size_t> results;
  Timer::Start("testing");
  nbc.Classify(testingData, results);
  Timer::Stop("testing");

  // Output results.
  const string outputFilename = CLI::GetParam<string>("output");
  data::Save(outputFilename.c_str(), results, true);

  return 0;
}
