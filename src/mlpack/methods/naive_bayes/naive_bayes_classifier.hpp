/**
 * @file naive_bayes_classifier.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  It is assumed that the features have been sampled from a
 * Gaussian PDF.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP
#define __MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/phi.hpp>

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
 * each of the labels y_j.  Alongwith this, it also computes the classs
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
 private:
  //! Sample mean for each class.
  MatType means;

  //! Sample variances for each class.
  MatType variances;

  //! Class probabilities.
  arma::vec probabilities;

 public:
  /**
   * Initializes the classifier as per the input and then trains it by
   * calculating the sample mean and variances.  The input data is expected to
   * have integer labels as the last row (starting with 0 and not greater than
   * the number of classes).
   *
   * Example use:
   * @code
   * extern arma::mat training_data, testing_data;
   * extern arma::Col<size_t> labels;
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
                       const arma::Col<size_t>& labels,
                       const size_t classes,
                       const bool incrementalVariance = false);

  /**
   * Given a bunch of data points, this function evaluates the class of each of
   * those data points, and puts it in the vector 'results'.
   *
   * @code
   * arma::mat test_data; // each column is a test point
   * arma::Col<size_t> results;
   * ...
   * nbc.Classify(test_data, &results);
   * @endcode
   *
   * @param data List of data points.
   * @param results Vector that class predictions will be placed into.
   */
  void Classify(const MatType& data, arma::Col<size_t>& results);

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
};

}; // namespace naive_bayes
}; // namespace mlpack

// Include implementation.
#include "naive_bayes_classifier_impl.hpp"

#endif
