/**
 * @file logistic_regression_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegressionFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP
#define MLPACK_METHODS_LOGISTIC_REGRESSION_FUNCTION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression_function.hpp"

namespace mlpack {
namespace regression {

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const double lambda)
  : // We promise to be well-behaved... the elements won't be modified.
    predictors(math::MakeAlias(const_cast<MatType&>(predictors), false)),
    responses(
        math::MakeAlias(const_cast<arma::Row<size_t>&>(responses), false)),
    lambda(lambda)
{
  initialPoint = arma::rowvec(predictors.n_rows + 1, arma::fill::zeros);

  // Sanity check.
  if (responses.n_elem != predictors.n_cols)
  {
    Log::Fatal << "LogisticRegressionFunction::LogisticRegressionFunction(): "
               << "predictors matrix has " << predictors.n_cols
               << " points, but "
               << "responses vector has " << responses.n_elem
               << " elements (should be"
               << " " << predictors.n_cols << ")!" << std::endl;
  }
}

template<typename MatType>
LogisticRegressionFunction<MatType>::LogisticRegressionFunction(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const arma::vec& initialPoint,
    const double lambda)
  : initialPoint(initialPoint),
    // We promise to be well-behaved... the elements won't be modified.
    predictors(math::MakeAlias(const_cast<MatType&>(predictors), false)),
    responses(
        math::MakeAlias(const_cast<arma::Row<size_t>&>(responses), false)),
    lambda(lambda)
{
  // To check if initialPoint is compatible with predictors.
  if (initialPoint.n_rows != (predictors.n_rows + 1)
      || initialPoint.n_cols != 1)
    this->initialPoint = arma::rowvec(predictors.n_rows + 1, arma::fill::zeros);
}

/**
 * Shuffle the datapoints.
 */
template<typename MatType>
void LogisticRegressionFunction<MatType>::Shuffle()
{
  MatType newPredictors;
  arma::Row<size_t> newResponses;

  math::ShuffleData(predictors, responses, newPredictors, newResponses);

  // If we are an alias, make sure we don't write to the original data.
  math::ClearAlias(predictors);
  math::ClearAlias(responses);

  // Take ownership of the new data.
  predictors = std::move(newPredictors);
  responses = std::move(newResponses);
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters.
 */
template<typename MatType>
double
LogisticRegressionFunction<MatType>::Evaluate(const arma::mat& parameters) const
{
  // The objective function is the log-likelihood function (w is the parameters
  // vector for the model; y is the responses; x is the predictors; sig() is the
  // sigmoid function):
  //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
  // We want to minimize this function.  L2-regularization is just lambda
  // multiplied by the squared l2-norm of the parameters then divided by two.

  // For the regularization, we ignore the first term, which is the intercept
  // term and take every term except the last one in the decision variable.
  const double regularization =
      0.5 * lambda * arma::dot(parameters.tail_cols(parameters.n_elem - 1),
                               parameters.tail_cols(parameters.n_elem - 1));

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const arma::rowvec exponents =
      parameters(0, 0)
      + parameters.tail_cols(parameters.n_elem - 1) * predictors;
  const arma::rowvec sigmoid = 1.0 / (1.0 + arma::exp(-exponents));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.
  double result = 0.0;
  for (size_t i = 0; i < responses.n_elem; ++i)
  {
    if (responses[i] == 1)
      result += log(sigmoid[i]);
    else
      result += log(1.0 - sigmoid[i]);
  }

  // Invert the result, because it's a minimization.
  return -result + regularization;
}

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters for a given batch from a given point.
 */
template<typename MatType>
double
LogisticRegressionFunction<MatType>::Evaluate(const arma::mat& parameters,
                                              const size_t begin,
                                              const size_t batchSize) const
{
  // Calculating the regularization term.
  const double regularization =
      lambda * (batchSize / (2.0 * predictors.n_cols))
      * arma::dot(parameters.tail_cols(parameters.n_elem - 1),
                  parameters.tail_cols(parameters.n_elem - 1));

  // Calculating the hypothesis that has to be passed to the sigmoid function.
  const arma::rowvec exponents =
      parameters(0, 0)
      + parameters.tail_cols(parameters.n_elem - 1)
            * predictors.cols(begin, begin + batchSize - 1);
  // Calculating the sigmoid function values.
  const arma::rowvec sigmoid = 1.0 / (1.0 + arma::exp(-exponents));

  // Iterating for the given batch size from a given point
  double result = 0.0;
  for (size_t i = 0; i < batchSize; ++i)
  {
    if (responses[i + begin] == 1)
      result += log(sigmoid[i]);
    else
      result += log(1.0 - sigmoid[i]);
  }

  // Invert the result, because it's a minimization.
  return -result + regularization;
}

//! Evaluate the gradient of the logistic regression objective function.
template<typename MatType>
void LogisticRegressionFunction<MatType>::Gradient(const arma::mat& parameters,
                                                   arma::mat& gradient) const
{
  // Regularization term.
  arma::mat regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1);

  const arma::rowvec sigmoids =
      (1 / (1 + arma::exp(-parameters(0, 0)
                          - parameters.tail_cols(parameters.n_elem - 1)
                                * predictors)));

  gradient.set_size(arma::size(parameters));
  gradient[0] = -arma::accu(responses - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) =
      (sigmoids - responses) * predictors.t() + regularization;
}

//! Evaluate the gradient of the logistic regression objective function for a
//! given batch size.
template<typename MatType>
template<typename GradType>
void LogisticRegressionFunction<MatType>::Gradient(const arma::mat& parameters,
                                                   const size_t begin,
                                                   GradType& gradient,
                                                   const size_t batchSize) const
{
  // Regularization term.
  arma::mat regularization;
  regularization = lambda * parameters.tail_cols(parameters.n_elem - 1)
                   / predictors.n_cols * batchSize;

  const arma::rowvec exponents =
      parameters(0, 0)
      + parameters.tail_cols(parameters.n_elem - 1)
            * predictors.cols(begin, begin + batchSize - 1);
  // Calculating the sigmoid function values.
  const arma::rowvec sigmoids = 1.0 / (1.0 + arma::exp(-exponents));

  gradient.set_size(parameters.n_rows, parameters.n_cols);
  gradient[0] =
      -arma::accu(responses.subvec(begin, begin + batchSize - 1) - sigmoids);
  gradient.tail_cols(parameters.n_elem - 1) =
      arma::sum((responses.subvec(begin, begin + batchSize - 1) - sigmoids)
                    * -predictors.cols(begin, begin + batchSize - 1).t(),
                0)
      + regularization;
}

/**
 * Evaluate the partial gradient of the logistic regression objective
 * function with respect to the individual features in the parameter.
 */
template<typename MatType>
void LogisticRegressionFunction<MatType>::PartialGradient(
    const arma::mat& parameters, const size_t j, arma::sp_mat& gradient) const
{
  const arma::rowvec diffs =
      responses
      - (1 / (1 + arma::exp(-parameters(0, 0)
                            - parameters.tail_cols(parameters.n_elem - 1)
                                  * predictors)));

  gradient.set_size(arma::size(parameters));

  if (j == 0)
  {
    gradient[j] = -arma::accu(diffs);
  }
  else
  {
    gradient[j] =
        arma::dot(-predictors.row(j - 1), diffs) + lambda * parameters(0, j);
  }
}

} // namespace regression
} // namespace mlpack

#endif
