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
template<typename VecType>
arma::vec NaiveBayesClassifier<MatType>::JointLogLikelihood(
                                      const VecType& point) const
{
  // Check that the number of features in the test data is same as in the
  // training data.
  Log::Assert(point.n_rows == means.n_rows);

  arma::vec jll = arma::log(probabilities);
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

    // Calculate oint log likelihood as sum of logarithm 
    // to decrease floating point errors.
    jll(i) += (point.n_rows / -2.0 * log(2 * M_PI) - 0.5 *
        log(arma::det(arma::diagmat(variances.col(i)))) + exponent);
  }

  return jll;
}

template<typename MatType>
arma::mat NaiveBayesClassifier<MatType>::JointLogLikelihood(
                                      const MatType& data) const
{
  arma::mat jll(data.n_cols, means.n_cols);
  for (size_t i = 0; i < data.n_cols; ++i)
  {
    jll.row(i) = JointLogLikelihood(data.col(i)).t();
  }
  return jll;
}        

template<typename MatType>
template<typename VecType>
size_t NaiveBayesClassifier<MatType>::Classify(const VecType& point) const
{
  // find the label(class) with max joint log likelihood.
  arma::vec jll = JointLogLikelihood(point);
  arma::uword maxIndex = 0;
  jll.max(maxIndex);
  return maxIndex;
}

template<typename MatType>
template<typename VecType>
void NaiveBayesClassifier<MatType>::Classify(const VecType& point,
                                             size_t& prediction,
                                             arma::vec& probabilities) const
{
  // log(Prob(Y|X)) = Log(X|Y) + Log(Y) - Log(X);
  // JointLogLikelihood = Log(X|Y) + Log(Y)
  arma::vec jll = JointLogLikelihood(point);
  double logProbX = log(arma::accu(exp(jll))); // Log(X)
  jll -= logProbX;

  arma::uword maxIndex = 0;
  jll.max(maxIndex);
  prediction = maxIndex;
  probabilities = exp(jll); // log(exp(value)) == value
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(
                                 const MatType& data,
                                 arma::Row<size_t>& predictions) const
{
  predictions.set_size(data.n_cols);

  arma::mat jll = JointLogLikelihood(data);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::uword maxIndex = 0;
    jll.row(i).max(maxIndex);
    predictions[i] = maxIndex;
  }
}

template<typename MatType>
void NaiveBayesClassifier<MatType>::Classify(const MatType& data,
                arma::Row<size_t>& predictions,
                arma::mat& predictionProbs) const
{
  predictions.set_size(data.n_cols);

  arma::mat jll = JointLogLikelihood(data);
  arma::vec logProbX(data.n_cols); // log(Prob(X))
  for (size_t j = 0; j < data.n_cols; ++j)
    logProbX(j) = log(arma::accu(exp(jll.row(j))));

  jll -= arma::repmat(logProbX, 1, data.n_cols);

  predictionProbs = exp(jll);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    arma::uword maxIndex = 0;
    jll.row(i).max(maxIndex);
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
