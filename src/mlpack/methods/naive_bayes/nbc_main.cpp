/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file nbc_main.cpp
 *
 * This program runs the Simple Naive Bayes Classifier.
 *
 * This classifier does parametric naive bayes classification assuming that the
 * features are sampled from a Gaussian distribution.
 *
 * PARAMETERS TO BE INPUT:
 *
 * --train
 * This is the file that contains the training data.
 *
 * --classes
 * This is the number of classes present in the training data.
 *
 * --test
 * This file contains the data points which the trained classifier should
 * classify.
 *
 * --output
 * This file will contain the classes to which the corresponding data points in
 * the testing data.
 */
#include <mlpack/core.hpp>

#include "naive_bayes_classifier.hpp"

PARAM_INT_REQ("classes", "The number of classes present in the data.", "c")

PARAM_STRING_REQ("train", "A file containing the training set", "t");
PARAM_STRING_REQ("test", "A file containing the test set", "T");
PARAM_STRING("output", "The file in which the output of the test would "
    "be written, defaults to 'output.csv')", "o", "output.csv");

PROGRAM_INFO("Parametric Naive Bayes", "This program test drives the Parametric"
    " Naive Bayes Classifier assuming that the features are sampled from a "
    "Gaussian distribution.");

using namespace mlpack;
using namespace naive_bayes;

int main(int argc, char* argv[])
{
  CLI::ParseCommandLine(argc, argv);

  const char *training_data_filename =
      CLI::GetParam<std::string>("train").c_str();
  arma::mat training_data;
  data::Load(training_data_filename, training_data, true);

  const char *testing_data_filename =
      CLI::GetParam<std::string>("test").c_str();
  arma::mat testing_data;
  data::Load(testing_data_filename, testing_data, true);

  size_t number_of_classes_ = CLI::GetParam<size_t>("classes");

  // Create and train the classifier.
  Timers::StartTimer("training");
  NaiveBayesClassifier<> nbc(training_data, number_of_classes_);
  Timers::StopTimer("training");

  // Timing the running of the Naive Bayes Classifier.
  arma::Col<size_t> results;
  Timers::StartTimer("testing");
  nbc.Classify(testing_data, results);
  Timers::StopTimer("testing");

  // Output results.
  std::string output_filename = CLI::GetParam<std::string>("output");
  data::Save(output_filename.c_str(), results, true);

  return 0;
}
