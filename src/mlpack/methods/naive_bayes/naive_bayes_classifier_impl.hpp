/**
 * @file naive_bayes_classifier_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Vahab Akbarzadeh (v.akbarzadeh@gmail.com)
 *
 * A Naive Bayes Classifier which parametrically estimates the distribution of
 * the features.  This classifier makes its predictions based on the assumption
 * that the features have been sampled from a set of Gaussians with diagonal
 * covariance.
 *
 * This file is part of MLPACK 1.0.10.
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
NaiveBayesClassifier<MatType>::NaiveBayesClassifier(
    const MatType& data,
    const arma::Col<size_t>& labels,
    const size_t classes,
    const bool incrementalVariance)
{
  const size_t dimensionality = data.n_rows;

  // Update the variables according to the number of features and classes
  // present in the data.
  probabilities.zeros(classes);
  means.zeros(dimensionality, classes);
  variances.zeros(dimensionality, classes);

  Log::Info << "Training Naive Bayes classifier on " << data.n_cols
      << " examples with " << dimensionality << " features each." << std::endl;

  // Calculate the class probabilities as well as the sample mean and variance
  // for each of the features with respect to each of the labels.
  if (incrementalVariance)
  {
    // Use incremental algorithm.
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      ++probabilities[label];

      arma::vec delta = data.col(j) - means.col(label);
      means.col(label) += delta / probabilities[label];
      variances.col(label) += delta % (data.col(j) - means.col(label));
    }

    for (size_t i = 0; i < classes; ++i)
    {
      if (probabilities[i] > 2)
        variances.col(i) /= (probabilities[i] - 1);
    }
  }
  else
  {
    // Don't use incremental algorithm.  This is a two-pass algorithm.  It is
    // possible to calculate the means and variances using a faster one-pass
    // algorithm but there are some precision and stability issues.  If this is
    // too slow, it's an option to use the faster algorithm by default and then
    // have this (and the incremental algorithm) be other options.

    // Calculate the means.
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      ++probabilities[label];
      means.col(label) += data.col(j);
    }

    // Normalize means.
    for (size_t i = 0; i < classes; ++i)
      if (probabilities[i] != 0.0)
        means.col(i) /= probabilities[i];

    // Calculate variances.
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      variances.col(label) += square(data.col(j) - means.col(label));
    }

    // Normalize variances.
    for (size_t i = 0; i < classes; ++i)
      if (probabilities[i] > 1)
        variances.col(i) /= (probabilities[i] - 1);
  }

  // Ensure that the variances are invertible.
  for (size_t i = 0; i < variances.n_elem; ++i)
    if (variances[i] == 0.0)
      variances[i] = 1e-50;

  probabilities /= data.n_cols;
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(const MatType& data,
                                             arma::Col<size_t>& results)
{
  // Check that the number of features in the test data is same as in the
  // training data.
  Log::Assert(data.n_rows == means.n_rows);

  arma::vec probs = arma::log(probabilities);
  arma::mat invVar = 1.0 / variances;

  arma::mat testProbs = arma::repmat(probs.t(), data.n_cols, 1);

  results.set_size(data.n_cols); // No need to fill with anything yet.

  Log::Info << "Running Naive Bayes classifier on " << data.n_cols
      << " data points with " << data.n_rows << " features each." << std::endl;

  // Calculate the joint probability for each of the data points for each of the
  // means.n_cols.

  // Loop over every class.
  for (size_t i = 0; i < means.n_cols; i++)
  {
    // This is an adaptation of gmm::phi() for the case where the covariance is
    // a diagonal matrix.
    arma::mat diffs = data - arma::repmat(means.col(i), 1, data.n_cols);
    arma::mat rhs = -0.5 * arma::diagmat(invVar.col(i)) * diffs;
    arma::vec exponents(diffs.n_cols);
    for (size_t j = 0; j < diffs.n_cols; ++j)
      exponents(j) = std::exp(arma::accu(diffs.col(j) % rhs.unsafe_col(j)));

    testProbs.col(i) += log(pow(2 * M_PI, (double) data.n_rows / -2.0) *
        pow(det(arma::diagmat(invVar.col(i))), -0.5) * exponents);
  }

  // Now calculate the label.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    // Find the index of the class with maximum probability for this point.
    arma::uword maxIndex = 0;
    arma::vec pointProbs = testProbs.row(i).t();
    pointProbs.max(maxIndex);

    results[i] = maxIndex;
  }

  return;
}

}; // namespace naive_bayes
}; // namespace mlpack

#endif
