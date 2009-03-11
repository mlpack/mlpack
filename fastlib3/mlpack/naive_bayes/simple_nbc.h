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
#ifndef NBC_H
#define NBC_H

#include "fastlib/fastlib.h"
#include "phi.h"
#include "math_functions.h"

const fx_entry_doc parm_nbc_entries[] ={
  {"training", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the training time\n"},
  {"testing", FX_TIMER, FX_CUSTOM, NULL,
   " The timer to record the testing time\n"},
  {"classes", FX_REQUIRED, FX_INT, NULL,
   " The number of classes present in the data\n"},
  {"features", FX_RESULT, FX_INT, NULL,
   " The number of features in the data\n"},
  {"examples", FX_RESULT, FX_INT, NULL,
   " The number of examples in the training set\n"},
  {"tests", FX_RESULT, FX_INT, NULL,
   " The number of data points in the test set\n"},
  FX_ENTRY_DOC_DONE
};

const fx_submodule_doc parm_nbc_submodules[] = {
  FX_SUBMODULE_DOC_DONE
};

const fx_module_doc parm_nbc_doc = {
  parm_nbc_entries, parm_nbc_submodules,
  " Trains the classifier using the training set and "
  "outputs the results for the test set\n"
};
  

/**
 * A classification class. The class labels are assumed
 * to be positive integers - 0,1,2,....
 *
 * This class trains on the data by calculating the 
 * sample mean and variance of the features with 
 * respect to each of the labels, and also the class
 * probabilities.
 *
 * Mathematically, it computes P(X_i = x_i | Y = y_j)
 * for each feature X_i for each of the labels y_j.
 * Alongwith this, it also computes the classs probabilities
 * P( Y = y_j)
 *
 * For classifying a data point (x_1, x_2, ..., x_n),
 * it computes the following:
 * arg max_y(P(Y = y)*P(X_1 = x_1 | Y = y) * ... * P(X_n = x_n | Y = y))
 *
 * Example use:
 * 
 * @code
 * SimpleNaiveBayesClassifier nbc;
 * Matrix training_data, testing_data;
 * datanode *nbc_module = fx_submodule(NULL,"nbc","nbc");
 * Vector results;
 * 
 * nbc.InitTrain(training_data, nbc_module);
 * nbc.Classify(testing_data, &results);
 * @endcode
 */
class SimpleNaiveBayesClassifier {

  // The class for testing this class is made a friend class
  friend class TestClassSimpleNBC;

 private:

  // The variables containing the sample mean and variance
  // for each of the features with respect to each class
  Matrix means_, variances_;

  // The variable containing the class probabilities
  ArrayList<double> class_probabilities_;

  // The variable keeping the information about the 
  // number of classes present
  index_t number_of_classes_;

  datanode *nbc_module_;
		   
 public:

  SimpleNaiveBayesClassifier(){
    means_.Init(0, 0);
    variances_.Init(0, 0);
    class_probabilities_.Init(0);
  }

  ~SimpleNaiveBayesClassifier(){
  }

 /**
  * The function that initializes the classifier as per the input
  * and then trains it by calculating the sample mean and variances
  *
  * Example use:
  * @code
  * Matrix training_data, testing_data;
  * datanode *nbc_module = fx_submodule(NULL,"nbc","nbc");
  * ....
  * nbc.InitTrain(training_data, nbc_module);
  * @endcode
  */
  void InitTrain(const Matrix& data, datanode* nbc_module) {

    ArrayList<double> feature_sum, feature_sum_squared;
    index_t number_examples = data.n_cols();
    index_t number_features = data.n_rows() - 1;
    nbc_module_ = nbc_module;

    // updating the variables, private and local, according to
    // the number of features and classes present in the data
    number_of_classes_ = fx_param_int_req(nbc_module_,"classes");
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
      class_probabilities_[i] = (double)number_of_occurrences 
	/ (double)number_examples ;
      for(index_t k = 0; k < number_features; k++) {
	means_.set(k, i, (feature_sum[k] / number_of_occurrences));
	variances_.set(k, i, (feature_sum_squared[k] 
			      - (feature_sum[k] * feature_sum[k] / number_of_occurrences))
			     /(number_of_occurrences - 1));
	feature_sum[k] = 0;
	feature_sum_squared[k] = 0;
      }
    }
  }

  /**
   * Given a bunch of data points, this function evaluates the class
   * of each of those data points, and puts it in the vector 'results'
   *
   * @code
   * Matrix test_data; // each column is a test point
   * Vector results;
   * ...
   * nbc.Classify(test_data, &results);
   * @endcode
   */
  void Classify(const Matrix& test_data, Vector *results){

    // Checking that the number of features in the test data is same
    // as in the training data
    DEBUG_ASSERT(test_data.n_rows() - 1 == means_.n_rows());

    ArrayList<double> tmp_vals;
    double *evaluated_result;
    index_t number_features = test_data.n_rows() - 1;
			
    evaluated_result = (double*)malloc(test_data.n_cols() * sizeof(double));
    tmp_vals.Init(number_of_classes_);
    
    NOTIFY("%"LI"d test cases with %"LI"d features each\n",
	   test_data.n_cols(), number_features);

    fx_result_int(nbc_module_,"tests", test_data.n_cols());
    // Calculating the joint probability for each of the data points
    // for each of the classes

    // looping over every test case
    for (index_t n = 0; n < test_data.n_cols(); n++) {			
      
      //looping over every class
      for (index_t i = 0; i < number_of_classes_; i++) {
	// Using the log values to prevent floating point underflow
	tmp_vals[i] = log(class_probabilities_[i]);
	for (index_t j = 0; j < number_features; j++) {
	  tmp_vals[i] += log(phi(test_data.get(j, n),
				 means_.get(j, i),
				 variances_.get(j, i))
			     );	  
	}
      }			
      // Calling a function 'max_element_index' from the file 'math_functions.h
      // to obtain the index of the maximum element in an array
      evaluated_result[n] = (double) max_element_index(tmp_vals);      
    }
    // The result is being put in a vector
    results->Copy(evaluated_result, test_data.n_cols());
    
    return;
  }
};
#endif
