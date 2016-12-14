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
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_IMPL_HPP
#define MLPACK_METHODS_NAIVE_BAYES_NAIVE_BAYES_CLASSIFIER_IMPL_HPP

#include <mlpack/core.hpp>

// In case it hasn't been included already.
#include "naive_bayes_classifier.hpp"

namespace mlpack {
namespace naive_bayes {

template<typename MatType>
NaiveBayesClassifier<MatType>::NaiveBayesClassifier(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t classes,
    const bool incremental) :
    trainingPoints(0) // Set when we call Train().
{
  const size_t dimensionality = data.n_rows;

  // Perform training, after initializing the model to 0 (that is, if Train()
  // won't do that for us, which it won't if we're using the incremental
  // algorithm).
  if (incremental)
  {
    probabilities.zeros(classes);
    means.zeros(dimensionality, classes);
    variances.zeros(dimensionality, classes);
  }
  else
  {
    probabilities.set_size(classes);
    means.set_size(dimensionality, classes);
    variances.set_size(dimensionality, classes);
  }
  Train(data, labels, incremental);
}

template<typename MatType>
NaiveBayesClassifier<MatType>::NaiveBayesClassifier(const size_t dimensionality,
                                                    const size_t classes) :
    trainingPoints(0)
{
  // Initialize model to 0.
  probabilities.zeros(classes);
  means.zeros(dimensionality, classes);
  variances.zeros(dimensionality, classes);
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Train(const MatType& data,
                                          const arma::Row<size_t>& labels,
                                          const bool incremental)
{
  // Calculate the class probabilities as well as the sample mean and variance
  // for each of the features with respect to each of the labels.
  if (incremental)
  {
    // Use incremental algorithm.
    // Fist, de-normalize probabilities.
    probabilities *= trainingPoints;

    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      ++probabilities[label];

      arma::vec delta = data.col(j) - means.col(label);
      means.col(label) += delta / probabilities[label];
      variances.col(label) += delta % (data.col(j) - means.col(label));
    }

    for (size_t i = 0; i < probabilities.n_elem; ++i)
    {
      if (probabilities[i] > 2)
        variances.col(i) /= (probabilities[i] - 1);
    }
  }
  else
  {
    // Set all parameters to zero
    probabilities.zeros();
    means.zeros();
    variances.zeros();

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
    for (size_t i = 0; i < probabilities.n_elem; ++i)
      if (probabilities[i] != 0.0)
        means.col(i) /= probabilities[i];

    // Calculate variances.
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t label = labels[j];
      variances.col(label) += square(data.col(j) - means.col(label));
    }

    // Normalize variances.
    for (size_t i = 0; i < probabilities.n_elem; ++i)
      if (probabilities[i] > 1)
        variances.col(i) /= (probabilities[i] - 1);
  }

  // Ensure that the variances are invertible.
  for (size_t i = 0; i < variances.n_elem; ++i)
    if (variances[i] == 0.0)
      variances[i] = 1e-50;

  probabilities /= data.n_cols;
  trainingPoints += data.n_cols;
}

template<typename MatType>
template<typename VecType>
void NaiveBayesClassifier<MatType>::Train(const VecType& point,
                                          const size_t label)
{
  // We must use the incremental algorithm here.
  probabilities *= trainingPoints;
  probabilities[label]++;

  arma::vec delta = point - means.col(label);
  means.col(label) += delta / probabilities[label];
  if (probabilities[label] > 2)
    variances.col(label) *= (probabilities[label] - 2);
  variances.col(label) += (delta % (point - means.col(label)));
  if (probabilities[label] > 1)
    variances.col(label) /= probabilities[label] - 1;

  trainingPoints++;
  probabilities /= trainingPoints;
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(const MatType& data,
                                             arma::Row<size_t>& results)
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
    for (size_t j = 0; j < diffs.n_cols; ++j) // log(exp(value)) == value
      exponents(j) = arma::accu(diffs.col(j) % rhs.unsafe_col(j));

    // Calculate probability as sum of logarithm to decrease floating point
    // errors.
    testProbs.col(i) += (data.n_rows / -2.0 * log(2 * M_PI) - 0.5 *
        log(arma::det(arma::diagmat(variances.col(i)))) + exponents);
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

template<typename MatType>
template<typename Archive>
void NaiveBayesClassifier<MatType>::Serialize(Archive& ar,
                                              const unsigned int /* version */)
{
  ar & data::CreateNVP(means, "means");
  ar & data::CreateNVP(variances, "variances");
  ar & data::CreateNVP(probabilities, "probabilities");
}

} // namespace naive_bayes
} // namespace mlpack

#endif
