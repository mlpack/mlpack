/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
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
 * --train 
 * This is the file that contains the training data
 *
 * --nbc/classes
 * This is the number of classes present in the training data
 *
 * --test
 * This file contains the data points which the trained
 * classifier would classify
 *
 * --output
 * This file will contain the classes to which the corresponding
 * data points in the testing data 
 * 
 */
#include "simple_nbc.h"

#include <armadillo>
#include <fastlib/base/arma_compat.h>

const fx_entry_doc parm_nbc_main_entries[] = {
  {"train", FX_REQUIRED, FX_STR, NULL,
   " A file containing the training set\n"},
  {"test", FX_REQUIRED, FX_STR, NULL,
   " A file containing the test set\n"},
  {"output", FX_PARAM, FX_STR, NULL,
   " The file in which the output of the test would be "
   "written (defaults to 'output.csv')\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc parm_nbc_main_submodules[] = {
  {"nbc", &parm_nbc_doc,
   " Trains on a given set and number of classes and "
   "tests them on a given set\n"},
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc parm_nbc_main_doc = {
  parm_nbc_main_entries, parm_nbc_main_submodules,
  "This program test drives the Parametric Naive Bayes \n"
  "Classifier assuming that the features are sampled \n"
  "from a Gaussian distribution.\n"
};

int main(int argc, char* argv[]) {

  fx_module *root = fx_init(argc, argv, &parm_nbc_main_doc);

  ////// READING PARAMETERS AND LOADING DATA //////

  const char *training_data_filename = fx_param_str_req(root, "train");
  arma::mat training_data;
  data::Load(training_data_filename, training_data);

  const char *testing_data_filename = fx_param_str_req(root, "test");
  arma::mat testing_data;
  data::Load(testing_data_filename, testing_data);

  ////// SIMPLE NAIVE BAYES CLASSIFICATION ASSUMING THE DATA TO BE UNIFORMLY DISTRIBUTED //////

  ////// Declaration of an object of the class SimpleNaiveBayesClassifier
  SimpleNaiveBayesClassifier nbc;

  struct datanode* nbc_module = fx_submodule(root, "nbc");
  
  ////// Timing the training of the Naive Bayes Classifier //////
  fx_timer_start(nbc_module, "training");

  ////// Calling the function that trains the classifier
  nbc.InitTrain(training_data, nbc_module);

  fx_timer_stop(nbc_module, "training");

  ////// Timing the testing of the Naive Bayes Classifier //////
  ////// The variable that contains the result of the classification
  arma::vec results;

  fx_timer_start(nbc_module, "testing");

  ////// Calling the function that classifies the test data
  nbc.Classify(testing_data, &results);

  fx_timer_stop(nbc_module, "testing");

  ////// OUTPUT RESULTS //////

  const char *output_filename = fx_param_str(root, "output", "output.csv");

  FILE *output_file = fopen(output_filename, "w");

  ot::Print(results, output_file);

  fclose(output_file);

  fx_done(root);

  return 1;
}
