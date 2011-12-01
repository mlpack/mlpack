/**
 * @file simple_nbc.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  It is assumed that the features have been sampled from a
 * Gaussian PDF.
 */
#ifndef __MLPACK_METHODS_NBC_SIMPLE_NBC_HPP
#define __MLPACK_METHODS_NBC_SIMPLE_NBC_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/phi.hpp>

namespace mlpack {
namespace naive_bayes {

PARAM_INT_REQ("classes", "The number of classes present in the data.", "nbc");

PARAM_MODULE("nbc", "Trains the classifier using the training set "
    "and outputs the results for the test set.");

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
class SimpleNaiveBayesClassifier
{
 public:
  //! Sample mean for each class.
  arma::mat means_;

  //! Sample variances for each class.
  arma::mat variances_;

  //! Class probabilities.
  arma::vec class_probabilities_;

  //! The number of classes present.
  size_t number_of_classes_;

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
  SimpleNaiveBayesClassifier(const arma::mat& data);

  /**
   * Default constructor, you need to use the other one.
   */
  SimpleNaiveBayesClassifier();

  ~SimpleNaiveBayesClassifier() { }

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

}; // namespace naive_bayes
}; // namespace mlpack

#endif
