/**
 * @file naive_bayes_classifier.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  It is assumed that the features have been sampled from a
 * Gaussian PDF.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP
#define MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace naive_bayes /** The Naive Bayes Classifier. */ {

/**
 * The simple Naive Bayes classifier.  This class trains on the data by
 * calculating the sample mean and variance of the features with respect to each
 * of the labels, and also the class probabilities.  The class labels are
 * assumed to be positive integers (starting with 0), and are expected to be the
 * last row of the data input to the constructor.
 *
 * Mathematically, it computes P(X_i = x_i | Y = y_j) for each feature X_i for
 * each of the labels y_j.  Alongwith this, it also computes the class
 * probabilities P(Y = y_j).
 *
 * For classifying a data point (x_1, x_2, ..., x_n), it computes the following:
 * arg max_y(P(Y = y)*P(X_1 = x_1 | Y = y) * ... * P(X_n = x_n | Y = y))
 *
 * Example use:
 *
 * @code
 * extern arma::mat training_data, testing_data;
 * NaiveBayesClassifier<> nbc(training_data, 5);
 * arma::vec results;
 *
 * nbc.Classify(testing_data, results);
 * @endcode
 */
template<typename MatType = arma::mat>
class NaiveBayesClassifier
{
 public:
  /**
   * Initializes the classifier as per the input and then trains it by
   * calculating the sample mean and variances.
   *
   * Example use:
   * @code
   * extern arma::mat training_data, testing_data;
   * extern arma::Row<size_t> labels;
   * NaiveBayesClassifier nbc(training_data, labels, 5);
   * @endcode
   *
   * @param data Training data points.
   * @param labels Labels corresponding to training data points.
   * @param classes Number of classes in this classifier.
   * @param incrementalVariance If true, an incremental algorithm is used to
   *     calculate the variance; this can prevent loss of precision in some
   *     cases, but will be somewhat slower to calculate.
   */
  NaiveBayesClassifier(const MatType& data,
                       const arma::Row<size_t>& labels,
                       const size_t classes,
                       const bool incrementalVariance = false);

  /**
   * Initialize the Naive Bayes classifier without performing training.  All of
   * the parameters of the model will be initialized to zero.  Be sure to use
   * Train() before calling Classify(), otherwise the results may be
   * meaningless.
   */
  NaiveBayesClassifier(const size_t dimensionality = 0,
                       const size_t classes = 0);

  /**
   * Train the Naive Bayes classifier on the given dataset.  If the incremental
   * algorithm is used, the current model is used as a starting point (this is
   * the default).  If the incremental algorithm is not used, then the current
   * model is ignored and the new model will be trained only on the given data.
   * Note that even if the incremental algorithm is not used, the data must have
   * the same dimensionality and number of classes that the model was
   * initialized with.  If you want to change the dimensionality or number of
   * classes, either re-initialize or call Means(), Variances(), and
   * Probabilities() individually to set them to the right size.
   *
   * @param data The dataset to train on.
   * @param incremental Whether or not to use the incremental algorithm for
   *      training.
   */
  void Train(const MatType& data,
             const arma::Row<size_t>& labels,
             const bool incremental = true);

  /**
   * Train the Naive Bayes classifier on the given point.  This will use the
   * incremental algorithm for updating the model parameters.  The data must be
   * the same dimensionality as the existing model parameters.
   *
   * @param point Data point to train on.
   * @param label Label of data point.
   */
  template<typename VecType>
  void Train(const VecType& point, const size_t label);

  /**
   * Given a bunch of data points, this function evaluates the class of each of
   * those data points, and puts it in the vector 'results'.
   *
   * @code
   * arma::mat test_data; // each column is a test point
   * arma::Row<size_t> results;
   * ...
   * nbc.Classify(test_data, &results);
   * @endcode
   *
   * @param data List of data points.
   * @param results Vector that class predictions will be placed into.
   */
  void Classify(const MatType& data, arma::Row<size_t>& results);

  //! Get the sample means for each class.
  const MatType& Means() const { return means; }
  //! Modify the sample means for each class.
  MatType& Means() { return means; }

  //! Get the sample variances for each class.
  const MatType& Variances() const { return variances; }
  //! Modify the sample variances for each class.
  MatType& Variances() { return variances; }

  //! Get the prior probabilities for each class.
  const arma::vec& Probabilities() const { return probabilities; }
  //! Modify the prior probabilities for each class.
  arma::vec& Probabilities() { return probabilities; }

  //! Serialize the classifier.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int /* version */);

 private:
  //! Sample mean for each class.
  MatType means;
  //! Sample variances for each class.
  MatType variances;
  //! Class probabilities.
  arma::vec probabilities;
  //! Number of training points seen so far.
  size_t trainingPoints;
};

} // namespace naive_bayes
} // namespace mlpack

// Include implementation.
#include "naive_bayes_classifier_impl.hpp"

#endif
