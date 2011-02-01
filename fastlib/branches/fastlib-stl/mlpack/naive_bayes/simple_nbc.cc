/**
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @file simple_nbc.h
 *
 * A Naive Bayes Classifier which parametrically 
 * estimates the distribution of the features.
 * It is assumed that the features have been 
 * sampled from a Gaussian PDF
 *
 */

#define ARMA_NO_DEBUG

#include "fastlib/fastlib.h"
#include "simple_nbc.h"
#include "phi.h"

void SimpleNaiveBayesClassifier::InitTrain(const arma::mat& data, datanode* nbc_module) {


  index_t number_examples = data.n_cols;
  index_t number_features = data.n_rows - 1;
  nbc_module_ = nbc_module;

  arma::vec feature_sum(number_features), feature_sum_squared(number_features);

  // updating the variables, private and local, according to
  // the number of features and classes present in the data
  number_of_classes_ = fx_param_int_req(nbc_module_,"classes");
  class_probabilities_.set_size(number_of_classes_);
  means_.set_size(number_features,number_of_classes_);
  variances_.set_size(number_features,number_of_classes_);

  NOTIFY("%"LI"d examples with %"LI"d features each\n",
	 number_examples, number_features);
  fx_result_int(nbc_module_, "features", number_features);
  fx_result_int(nbc_module_, "examples", number_examples);

  // calculating the class probabilities as well as the 
  // sample mean and variance for each of the features
  // with respect to each of the labels
  for(index_t i = 0; i < number_of_classes_; i++ ) {
    index_t number_of_occurrences = 0;
    for (index_t j = 0; j < number_examples; j++) {
      index_t flag = (index_t)  data(number_features, j);
      if(i == flag) {
	++number_of_occurrences;
	for(index_t k = 0; k < number_features; k++) {
	  double tmp = data(k, j);
	  feature_sum(k) += tmp;
	  feature_sum_squared(k) += tmp*tmp;
	}
      }
    }
    class_probabilities_[i] = (double)number_of_occurrences 
      / (double)number_examples ;
    for(index_t k = 0; k < number_features; k++) {
      double fs = feature_sum(k),
	     fss = feature_sum_squared(k);

      means_(k, i) = (fs / number_of_occurrences);
      variances_(k, i) = (fss 
			    - (fs * fs / number_of_occurrences))
			   /(number_of_occurrences - 1);
      /*
      means_(k, i) = (feature_sum(k) / number_of_occurrences);
      variances_(k, i) = (feature_sum_squared(k) 
			    - (feature_sum(k) * feature_sum(k) / number_of_occurrences))
			   /(number_of_occurrences - 1);
       */
    }
    feature_sum.zeros(number_features);
    feature_sum_squared.zeros(number_features);
  }
}

void SimpleNaiveBayesClassifier::Classify(const arma::mat& test_data, arma::vec& results){

  // Checking that the number of features in the test data is same
  // as in the training data
  DEBUG_ASSERT(test_data.n_rows - 1 == means_.n_rows);

  arma::vec tmp_vals(number_of_classes_);
  index_t number_features = test_data.n_rows - 1;
		      
  results.zeros(test_data.n_cols);
  
  NOTIFY("%"LI"d test cases with %"LI"d features each\n",
	 test_data.n_cols, number_features);

  fx_result_int(nbc_module_,"tests", test_data.n_cols);
  // Calculating the joint probability for each of the data points
  // for each of the classes

  // looping over every test case
  for (index_t n = 0; n < test_data.n_cols; n++) {			
    
    //looping over every class
    for (index_t i = 0; i < number_of_classes_; i++) {
      // Using the log values to prevent floating point underflow
      tmp_vals(i) = log(class_probabilities_(i));

      //looping over every feature
      for (index_t j = 0; j < number_features; j++) {
	tmp_vals(i) += log(phi(test_data(j, n),
			       means_(j, i),
			       variances_(j, i))
			   );	  
      }
    }			

    // Find the index of the maximum value in tmp_vals.
    index_t max = 0;
    for (index_t k = 0; k < number_of_classes_; k++) {
      if(tmp_vals(max) < tmp_vals(k))
	max = k;
    }
    results(n) = max;
  }
  
  return;
}
