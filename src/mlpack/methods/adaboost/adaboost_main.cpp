/*
 * @file: adaboost_main.cpp
 * @author: Udit Saxena
 *
 *
 */

#include <mlpack/core.hpp>
#include "adaboost.hpp"

using namespace mlpack;
using namespace std;
using namespace arma;
using namespace mlpack::adaboost;

PROGRAM_INFO("","");

//necessary parameters
PARAM_STRING_REQ("train_file", "A file containing the training set.", "t");
PARAM_STRING_REQ("labels_file", "A file containing labels for the training set.",
  "l");
PARAM_STRING_REQ("test_file", "A file containing the test set.", "T");

//optional parameters.
PARAM_STRING("output", "The file in which the predicted labels for the test set"
    " will be written.", "o", "output.csv");
PARAM_INT("iterations","The maximum number of boosting iterations "
  "to be run", "i", 1000);
PARAM_INT_REQ("classes","The number of classes in the input label set.","c");

int main(int argc, char *argv[])
{
  CLI::ParseCommandLine(argc, argv);
  
  const string trainingDataFilename = CLI::GetParam<string>("train_file");
  mat trainingData;
  data::Load(trainingDataFilename, trainingData, true);
  
  const string labelsFilename = CLI::GetParam<string>("labels_file");
  // Load labels.
  mat labelsIn;
  // data::Load(labelsFilename, labelsIn, true);

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
  
  int classes = 3;
  
  // define your own weak learner, perceptron in this case.
  int iter = 4000;
  
  perceptron::Perceptron<> p(trainingData, labels.t(), iter);
  
  Timer::Start("Training");
  Adaboost<> a(trainingData, labels.t(), iterations, classes, p);
  Timer::Stop("Training");

  // vec results;
  // data::RevertLabels(predictedLabels, mappings, results);

  // const string outputFilename = CLI::GetParam<string>("output");
  // data::Save(outputFilename, results, true, true);

  return 0;
}