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
 * arma::mat training_data, testing_data;
 * datanode *nbc_module = fx_submodule(NULL,"nbc","nbc");
 * arma::vec results;
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
  arma::mat means_, variances_;

  // The variable containing the class probabilities
  arma::vec class_probabilities_;

  // The variable keeping the information about the 
  // number of classes present
  index_t number_of_classes_;

  datanode* nbc_module_;
		   
 public:

 /**
  * Initializes the classifier as per the input and then trains it
  * by calculating the sample mean and variances
  *
  * Example use:
  * @code
  * arma::mat training_data, testing_data;
  * datanode nbc_module = fx_submodule(NULL,"nbc","nbc");
  * ....
  * SimpleNaiveBayesClassifier nbc(training_data, nbc_module);
  * @endcode
  */
  SimpleNaiveBayesClassifier(const arma::mat& data, datanode* nbc_module);
  /**
   * Default constructor, you need to use the other one.
  */
  SimpleNaiveBayesClassifier();
  ~SimpleNaiveBayesClassifier(){
  }

  /**
   * Given a bunch of data points, this function evaluates the class
   * of each of those data points, and puts it in the vector 'results'
   *
   * @code
   * arma::mat test_data; // each column is a test point
   * arma::vec results;
   * ...
   * nbc.Classify(test_data, &results);
   * @endcode
   */
  void Classify(const arma::mat& test_data, arma::vec& results);
};
#endif
