/*
 * @file: perceptron_main.cpp
 * @author: Udit Saxena
 *
 *
 */

#include <mlpack/core.hpp>
#include "perceptron.hpp"

using namespace mlpack;
using namespace mlpack::perceptron;
using namespace std;
using namespace arma;

PROGRAM_INFO("","");

//necessary parameters
PARAM_STRING_REQ("train_file", "A file containing the training set.", "tr");
PARAM_STRING_REQ("labels_file", "A file containing labels for the training set.",
  "l");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "te");

//optional parameters.
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
  data::Load(labelsFilename, labelsIn, true);

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