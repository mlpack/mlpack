/**
 * @file naive_bayes_classifier_impl.hpp
 * @author Parikshit Ram (pram@cc.gatech.edu)
 * @author Vahab Akbarzadeh (v.akbarzadeh@gmail.com)
 * @author Shihao Jing (shihao.jing810@gmail.com)
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

#include <mlpack/prereqs.hpp>

// In case it hasn't been included already.
#include "naive_bayes_classifier.hpp"

namespace mlpack {
namespace naive_bayes {

template<typename MatType>
NaiveBayesClassifier<MatType>::NaiveBayesClassifier(
    const MatType& data,
    const arma::Row<size_t>& labels,
    const size_t numClasses,
    const bool incremental) :
    trainingPoints(0) // Set when we call Train().
{
  if (incremental)
  {
    probabilities.zeros(numClasses);
    means.zeros(data.n_rows, numClasses);
    variances.zeros(data.n_rows, numClasses);
  }
  else
  {
    probabilities.set_size(numClasses);
    means.set_size(data.n_rows, numClasses);
    variances.set_size(data.n_rows, numClasses);
  }

  Train(data, labels, numClasses, incremental);
}

template<typename MatType>
NaiveBayesClassifier<MatType>::NaiveBayesClassifier(const size_t dimensionality,
                                                    const size_t numClasses) :
    trainingPoints(0)
{
  // Initialize model to 0.
  probabilities.zeros(numClasses);
  means.zeros(dimensionality, numClasses);
  variances.zeros(dimensionality, numClasses);
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Train(const MatType& data,
                                          const arma::Row<size_t>& labels,
                                          const size_t numClasses,
                                          const bool incremental)
{
  // Do we need to resize the model?
  if (probabilities.n_elem != numClasses)
  {
    // Perform training, after initializing the model to 0 (that is, if Train()
    // won't do that for us, which it won't if we're using the incremental
    // algorithm).
    if (incremental)
    {
      probabilities.zeros(numClasses);
      means.zeros(data.n_rows, numClasses);
      variances.zeros(data.n_rows, numClasses);
    }
    else
    {
      probabilities.set_size(numClasses);
      means.set_size(data.n_rows, numClasses);
      variances.set_size(data.n_rows, numClasses);
    }
  }

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
    // Set all parameters to zero.
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
template<typename VecType>
void NaiveBayesClassifier<MatType>::LogLikelihood(
    const VecType& point,
    arma::vec& logLikelihoods) const
{
  // Check that the number of features in the test data is same as in the
  // training data.
  Log::Assert(point.n_rows == means.n_rows);

  logLikelihoods = arma::log(probabilities);
  arma::mat invVar = 1.0 / variances;

  // Calculate the joint log likelihood of point for each of the
  // means.n_cols.

  // Loop over every class.
  for (size_t i = 0; i < means.n_cols; i++)
  {
    // This is an adaptation of gmm::phi() for the case where the covariance is
    // a diagonal matrix.
    arma::vec diffs = point - means.col(i);
    arma::vec rhs = -0.5 * arma::diagmat(invVar.col(i)) * diffs;
    double exponent = arma::accu(diffs % rhs); // log(exp(value)) == value

    // Calculate point log likelihood as sum of logs to decrease floating point
    // errors.
    logLikelihoods(i) += (point.n_rows / -2.0 * log(2 * M_PI) - 0.5 *
        log(arma::det(arma::diagmat(variances.col(i)))) + exponent);
  }
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::LogLikelihood(
    const MatType& data,
    arma::mat& logLikelihoods) const
{
  logLikelihoods.set_size(means.n_cols, data.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::vec v = logLikelihoods.unsafe_col(i);
    LogLikelihood(data.col(i), v);
  }
}

template<typename MatType>
template<typename VecType>
size_t NaiveBayesClassifier<MatType>::Classify(const VecType& point) const
{
  // Find the label(class) with max log likelihood.
  arma::vec logLikelihoods;
  LogLikelihood(point, logLikelihoods);

  arma::uword maxIndex = 0;
  logLikelihoods.max(maxIndex);
  return maxIndex;
}

template<typename MatType>
template<typename VecType>
void NaiveBayesClassifier<MatType>::Classify(const VecType& point,
                                             size_t& prediction,
                                             arma::vec& probabilities) const
{
  // log(Prob(Y|X)) = Log(Prob(X|Y)) + Log(Prob(Y)) - Log(Prob(X));
  // But LogLikelihood() gives us the unnormalized log likelihood which is
  // Log(Prob(X|Y)) + Log(Prob(Y)) so we need to subtract the normalization
  // term.
  arma::vec logLikelihoods;
  LogLikelihood(point, logLikelihoods);
  const double logProbX = log(arma::accu(exp(logLikelihoods))); // Log(Prob(X)).
  logLikelihoods -= logProbX;

  arma::uword maxIndex = 0;
  logLikelihoods.max(maxIndex);
  prediction = (size_t) maxIndex;
  probabilities = exp(logLikelihoods); // log(exp(value)) == value.
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);

  arma::mat logLikelihoods;
  LogLikelihood(data, logLikelihoods);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::uword maxIndex = 0;
    logLikelihoods.unsafe_col(i).max(maxIndex);
    predictions[i] = maxIndex;
  }
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(
    const MatType& data,
    arma::Row<size_t>& predictions,
    arma::mat& predictionProbs) const
{
  predictions.set_size(data.n_cols);

  arma::mat logLikelihoods;
  LogLikelihood(data, logLikelihoods);

  arma::vec logProbX(data.n_cols); // log(Prob(X)) for each point.
  for (size_t j = 0; j < data.n_cols; ++j)
  {
    logProbX(j) = log(arma::accu(exp(logLikelihoods.col(j))));
    logLikelihoods.col(j) -= logProbX(j);
  }

  predictionProbs = arma::exp(logLikelihoods);

  // Now calculate maximum probabilities for each point.
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::uword maxIndex;
    logLikelihoods.unsafe_col(i).max(maxIndex);
    predictions[i] = maxIndex;
  }
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
