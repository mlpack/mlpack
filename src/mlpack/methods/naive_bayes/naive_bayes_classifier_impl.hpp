/**
 * @file simple_nbc_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  It is assumed that the features have been sampled from a
 * Gaussian PDF.
 *
 * This file is part of MLPACK 1.0.5.
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
#ifndef __MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_IMPL_HPP
#define __MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_IMPL_HPP

#include <mlpack/core.hpp>

// In case it hasn't been included already.
#include "naive_bayes_classifier.hpp"

namespace mlpack {
namespace naive_bayes {

template<typename MatType>
NaiveBayesClassifier<MatType>::NaiveBayesClassifier(const MatType& data,
                                                    const size_t classes)
{
  size_t dimensionality = data.n_rows - 1;

  // Update the variables according to the number of features and classes
  // present in the data.
  probabilities.set_size(classes);
  means.zeros(dimensionality, classes);
  variances.zeros(dimensionality, classes);

  Log::Info << "Training Naive Bayes classifier on " << data.n_cols
      << " examples with " << dimensionality << " features each." << std::endl;

  // Calculate the class probabilities as well as the sample mean and variance
  // for each of the features with respect to each of the labels.
  for (size_t j = 0; j < data.n_cols; ++j)
  {
    size_t label = (size_t) data(dimensionality, j);
    ++probabilities[label];

    means.col(label) += data(arma::span(0, dimensionality - 1), j);
    variances.col(label) += square(data(arma::span(0, dimensionality - 1), j));
  }

  for (size_t i = 0; i < classes; ++i)
  {
    variances.col(i) -= (square(means.col(i)) / probabilities[i]);
    means.col(i) /= probabilities[i];
    variances.col(i) /= (probabilities[i] - 1);
  }

  probabilities /= data.n_cols;
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(const MatType& data,
                                             arma::Col<size_t>& results)
{
  // Check that the number of features in the test data is same as in the
  // training data.
  Log::Assert(data.n_rows == means.n_rows);

  arma::vec probs(means.n_cols);

  results.zeros(data.n_cols);

  Log::Info << "Running Naive Bayes classifier on " << data.n_cols
      << " data points with " << data.n_rows << " features each." << std::endl;

  // Calculate the joint probability for each of the data points for each of the
  // means.n_cols.

  // Loop over every test case.
  for (size_t n = 0; n < data.n_cols; n++)
  {
    // Loop over every class.
    for (size_t i = 0; i < means.n_cols; i++)
    {
      // Use the log values to prevent floating point underflow.
      probs(i) = log(probabilities(i));

      // Loop over every feature.
      probs(i) += log(gmm::phi(data.unsafe_col(n), means.unsafe_col(i),
          diagmat(variances.unsafe_col(i))));
    }

    // Find the index of the maximum value in tmp_vals.
    arma::uword maxIndex = 0;
    probs.max(maxIndex);

    results[n] = maxIndex;
  }

  return;
}

}; // namespace naive_bayes
}; // namespace mlpack

#endif
