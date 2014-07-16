/*
 * @file: perceptron_main.cpp
 * @author: Udit Saxena
 *
 * Main executable for the Perceptron.
 */

#include <mlpack/core.hpp>
#include "perceptron.hpp"

using namespace mlpack;
using namespace mlpack::perceptron;
using namespace std;
using namespace arma;

PROGRAM_INFO("Perceptron",
    "This program implements a perceptron, which is a single level "
    "Neural Network. The perceptron makes its predictions based on "
    "a linear predictor function combining a set of weights with the feature "
    "vector.\n"
    "The perceptron learning rule is able to converge, given enough iterations "
    "using the --iterations (-i) parameter, if the data supplied is "
    "linearly separable. "
    "\n"
    "The Perceptron is parameterized by a matrix of weight vectors which "
    "denotes the numerical weights of the Neural Network."
    "\n"
    "This program allows training of a perceptron, and then application of "
    "the learned perceptron to a test dataset.  To train a perceptron, "
    "a training dataset must be passed to --train_file (-t).  Labels can either "
    "be present as the last dimension of the training dataset, or given "
    "explicitly with the --labels_file (-l) parameter."
    "\n"
    "A test file is given through the --test_file (-T) parameter.  The "
    "predicted labels for the test set will be stored in the file specified by "
    "the --output_file (-o) parameter."
    );

// Necessary parameters
PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING("labels_file", "A file containing labels for the training set.",
  "l","");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");

// Optional parameters.
PARAM_STRING("output", "The file in which the predicted labels for the test set"
    " will be written.", "o", "output.csv");
PARAM_INT("iterations","The maximum number of iterations the perceptron is "
  "to be run", "i", 1000)

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);

  const string trainingDataFilename = CLI::GetParam<string>("train_file");
  mat trainingData;
  data::Load(trainingDataFilename, trainingData, true);

  const string labelsFilename = CLI::GetParam<string>("labels_file");
  // Load labels.
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
  // helpers for normalizing the labels
  Col<size_t> labels;
  vec mappings;

  // Do the labels need to be transposed?
  if (labelsIn.n_rows == 1)
    labelsIn = labelsIn.t();

  // normalize the labels
  data::NormalizeLabels(labelsIn.unsafe_col(0), labels, mappings);

  const string testingDataFilename = CLI::GetParam<string>("test_file");
  mat testingData;
  data::Load(testingDataFilename, testingData, true);

  if (testingData.n_rows != trainingData.n_rows)
    Log::Fatal << "Test data dimensionality (" << testingData.n_rows << ") "
        << "must be the same as training data (" << trainingData.n_rows - 1
        << ")!" << std::endl;
  int iterations = CLI::GetParam<int>("iterations");
  
  Timer::Start("Training");
  Perceptron<> p(trainingData, labels, iterations);
  Timer::Stop("Training");

  Row<size_t> predictedLabels(testingData.n_cols);
  Timer::Start("Testing");
  p.Classify(testingData, predictedLabels);
  Timer::Stop("Testing");

  vec results;
  data::RevertLabels(predictedLabels, mappings, results);

  const string outputFilename = CLI::GetParam<string>("output");
  data::Save(outputFilename, results, true, true);
  // saving the predictedLabels in the transposed manner in output

  return 0;
}