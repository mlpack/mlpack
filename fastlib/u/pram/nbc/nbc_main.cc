/**
 * @file nbc_main.cc
 * 
 * This program test drives the Simple Naive Bayes Classifier
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

  SimpleNaiveBayesClassifier nbc;

  struct datanode* nbc_module = fx_submodule(NULL, "simple_nbc", "simple_nbc_module");
  const int number_of_classes = fx_param_int_req(NULL, "number_of_classes");

  ////// Timing the training of the Naive Bayes Classifier //////
  fx_timer_start(nbc_module, "training_classifier");

  nbc.InitTrain(training_data, number_of_classes);

  fx_timer_stop(nbc_module, "training_classifier");

  ////// Timing the testing of the Naive Bayes Classifier //////
  Vector results;

  fx_timer_start(nbc_module, "testing_classifier");

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
