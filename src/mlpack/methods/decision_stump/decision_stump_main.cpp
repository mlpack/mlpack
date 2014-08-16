/**
 * @file decision_stump_main.cpp
 * @author Udit Saxena
 *
 * Main executable for the decision stump.
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
    "--bin_size (-b) parameter.\n"
    "\n"
    "The decision stump is parameterized by a splitting dimension and a vector "
    "of values that denote the splitting values of each bin.\n"
    "\n"
    "This program allows training of a decision stump, and then application of "
    "the learned decision stump to a test dataset.  To train a decision stump, "
    "a training dataset must be passed to --train_file (-t).  Labels can either"
    " be present as the last dimension of the training dataset, or given "
    "explicitly with the --labels_file (-l) parameter.\n"
    "\n"
    "A test file is given through the --test_file (-T) parameter.  The "
    "predicted labels for the test set will be stored in the file specified by "
    "the --output_file (-o) parameter.");

// Necessary parameters.
PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");

// Output parameters (optional).
PARAM_STRING("labels_file", "A file containing labels for the training set. If "
    "not specified, the labels are assumed to be the last row of the training "
    "data.", "l", "");
PARAM_STRING("output_file", "The file in which the predicted labels for the "
    "test set will be written.", "o", "output.csv");

PARAM_INT("bin_size", "The minimum number of training points in each "
    "decision stump bin.", "b", 6);

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  const string trainingDataFilename = CLI::GetParam<string>("train_file");
  mat trainingData;
  data::Load(trainingDataFilename, trainingData, true);

  // Load labels, if necessary.
  mat labelsIn;
  if (CLI::HasParam("labels_file"))
  {
    const string labelsFilename = CLI::GetParam<string>("labels_file");
    // Load labels.
    data::Load(labelsFilename, labelsIn, true);

    // Do the labels need to be transposed?
    if (labelsIn.n_rows == 1)
      labelsIn = labelsIn.t();
  }
  else
  {
    // Extract the labels as the last
    Log::Info << "Using the last dimension of training set as labels." << endl;

    labelsIn = trainingData.row(trainingData.n_rows - 1).t();
    trainingData.shed_row(trainingData.n_rows - 1);
  }

  // Normalize the labels.
  Col<size_t> labels;
  vec mappings;
  data::NormalizeLabels(labelsIn.unsafe_col(0), labels, mappings);

  const size_t inpBucketSize = CLI::GetParam<int>("bucket_size");
  const size_t numClasses = labels.max() + 1;

  // Load the test file.
  const string testingDataFilename = CLI::GetParam<std::string>("test_file");
  mat testingData;
  data::Load(testingDataFilename, testingData, true);

  if (testingData.n_rows != trainingData.n_rows)
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << trainingData.n_rows - 1
        << ")!" << std::endl;

  Timer::Start("training");
  DecisionStump<> ds(trainingData, labels.t(), numClasses,
                     inpBucketSize);
  Timer::Stop("training");

  Row<size_t> predictedLabels(testingData.n_cols);
  Timer::Start("testing");
  ds.Classify(testingData, predictedLabels);
  Timer::Stop("testing");

  vec results;
  data::RevertLabels(predictedLabels.t(), mappings, results);

  // Save the predicted labels in a transposed form as output.
  const string outputFilename = CLI::GetParam<string>("output_file");
  data::Save(outputFilename, results, true, false);
}
