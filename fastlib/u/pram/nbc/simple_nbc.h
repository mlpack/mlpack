/**
 * @file simple_nbc.h
 *
 * A Simple Naive Bayes Classifier assuming that the data
 * is generated from a gaussian distribution
 *
 */

#include "fastlib/fastlib.h"
#include "phi.h"
#include "math_functions.h"

class SimpleNaiveBayesClassifier {
  friend class TestClassSimpleNBC;
 private:
  Matrix means_, variances_;
  ArrayList<double> class_probabilities_;
  index_t number_of_classes_;
		   
 public:
  SimpleNaiveBayesClassifier(){
    means_.Init(0, 0);
    variances_.Init(0, 0);
    class_probabilities_.Init(0);
  }

  ~SimpleNaiveBayesClassifier(){
    means_.~Matrix();
    variances_.~Matrix();
    class_probabilities_.Clear();
    class_probabilities_.Destruct();
  }



  void InitTrain(const Matrix& data, int number_of_classes) {
    ArrayList<double> feature_sum, feature_sum_squared;
    /** the last row are the classes */
    index_t number_examples = data.n_cols(); // number of examples in the dataset
    index_t number_features = data.n_rows() - 1; // number of features in each example
    /** the classes are of the form 0,1,2...,n_classes - 1 */
    number_of_classes_ = number_of_classes;
    class_probabilities_.Resize(number_of_classes_);
    means_.Destruct();
    means_.Init(number_features, number_of_classes_ );
    variances_.Destruct();
    variances_.Init(number_features, number_of_classes_);
    feature_sum.Init(number_features);
    feature_sum_squared.Init(number_features);
    for(index_t k = 0; k < number_features; k++) {
      feature_sum[k] = 0;
      feature_sum_squared[k] = 0;
    }
    printf("%"LI"d examples with %"LI"d features each\n",number_examples, number_features);
    // calculating the probablity of occurrence of the individual classes
    for(index_t i = 0; i < number_of_classes_; i++ ) {
      index_t number_of_occurrences = 0;
      for (index_t j = 0; j < number_examples; j++) {
	index_t flag = (index_t)  data.get(number_features, j);
	if(i == flag) {
	  ++number_of_occurrences;
	  for(index_t k = 0; k < number_features; k++) {
	    double tmp = data.get(k, j);
	    feature_sum[k] += tmp;
	    feature_sum_squared[k] += tmp*tmp;
	  }
	}
      }
      class_probabilities_[i] = (double)number_of_occurrences / (double)number_examples ;
      for(index_t k = 0; k < number_features; k++) {
	means_.set(k, i, (feature_sum[k] / number_of_occurrences));
	variances_.set(k, i, (feature_sum_squared[k] - (feature_sum[k] * feature_sum[k] / number_of_occurrences)) / (number_of_occurrences - 1));
	feature_sum[k] = 0;
	feature_sum_squared[k] = 0;
      }
    }
  }

  void Classify(const Matrix& test_data, Vector *results){
    DEBUG_ASSERT(test_data.n_rows() - 1 == means_.n_rows());
    ArrayList<double> tmp_vals;
    double *evaluated_result;
    index_t number_features = test_data.n_rows() - 1;
			
    evaluated_result = (double*)malloc(test_data.n_cols() * sizeof(double));
    tmp_vals.Init(number_of_classes_);
    
    printf("%"LI"d test cases with %"LI"d features each\n", test_data.n_cols(), number_features);

    for (index_t n = 0; n < test_data.n_cols(); n++) {			
      for (index_t i = 0; i < number_of_classes_; i++) {
	tmp_vals[i] = log(class_probabilities_[i]);
	for (index_t j = 0; j < number_features; j++) {
	  tmp_vals[i] += log(phi(test_data.get(j, n), means_.get(j, i), variances_.get(j, i)));	  
	}
      }			
      evaluated_result[n] = (double) max_element_index(tmp_vals);      
    }
    (*results).Copy(evaluated_result, test_data.n_cols());
    
    return;
  }
};
