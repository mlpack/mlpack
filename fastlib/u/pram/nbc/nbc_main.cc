/**
 * @author pram
 * @file nbc_main.cc
 * 
 * This program test drives the Simple Naive Bayes Classifier
 * 
 * This classifier does parametric naive bayes classification
 * assuming that the features are sampled from a Gaussian
 * distribution.
 *
 * PARAMETERS TO BE INPUT:
 * 
 * --training_data 
 * This is the file that contains the training data
 *
 * --number_of_classes
 * This is the number of classes present in the training data
 *
 * --testing_data
 * This file contains the data points which the trained
 * classifier would classify
 *
 * --output_filename
 * This file will contain the classes to which the corresponding
 * data points in the testing data 
 * 
 */
#include "simple_nbc.h"


int main(int argc, char* argv[]) {

  fx_init(argc, argv);

  ////// READING PARAMETERS AND LOADING DATA //////

  const char *training_data_filename = fx_param_str_req(NULL, "training_data");
  Matrix training_data;
  data::Load(training_data_filename, &training_data);

  const char *testing_data_filename = fx_param_str_req(NULL, "testing_data");
  Matrix testing_data;
  data::Load(testing_data_filename, &testing_data);

  ////// SIMPLE NAIVE BAYES CLASSIFICATION ASSUMING THE DATA TO BE UNIFORMLY DISTRIBUTED //////

  ////// Declaration of an object of the class SimpleNaiveBayesClassifier
  SimpleNaiveBayesClassifier nbc;

  struct datanode* nbc_module = fx_submodule(NULL, "simple_nbc", "simple_nbc_module");
  const int number_of_classes = fx_param_int_req(NULL, "number_of_classes");

  ////// Timing the training of the Naive Bayes Classifier //////
  fx_timer_start(nbc_module, "training_classifier");

  ////// Calling the function that trains the classifier
  nbc.InitTrain(training_data, number_of_classes);

  fx_timer_stop(nbc_module, "training_classifier");

  ////// Timing the testing of the Naive Bayes Classifier //////
  ////// The variable that contains the result of the classification
  Vector results;

  fx_timer_start(nbc_module, "testing_classifier");

  ////// Calling the function that classifies the test data
  nbc.Classify(testing_data, &results);

  fx_timer_stop(nbc_module, "testing_classifier");

  ////// OUTPUT RESULTS //////

  const char *output_filename = fx_param_str(NULL, "output_filename", "output.csv");

  FILE *output_file = fopen(output_filename, "w");

  ot::Print(results, output_file);

  fclose(output_file);

  fx_done();

  return 0;
}
