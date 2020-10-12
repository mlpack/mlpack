/**
 * @file methods/logistic_regression/logistic_regression_impl.hpp
 * @author Sumedh Ghaisas
 * @author Arun Reddy
 *
 * Implementation of the LogisticRegression class.  This implementation supports
 * L2-regularization.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
#define MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression.hpp"

namespace mlpack {
namespace regression {

template<typename MatType>
LogisticRegression<MatType>::LogisticRegression(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const double lambda) :
    lambda(lambda)
{
  Train(predictors, responses);
}

template<typename MatType>
LogisticRegression<MatType>::LogisticRegression(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const arma::rowvec& initialPoint,
    const double lambda) :
    parameters(initialPoint),
    lambda(lambda)
{
  Train(predictors, responses);
}

template<typename MatType>
LogisticRegression<MatType>::LogisticRegression(
    const size_t dimensionality,
    const double lambda) :
    parameters(arma::rowvec(dimensionality + 1, arma::fill::zeros)),
    lambda(lambda)
{
  // No training to do here.
}

template<typename MatType>
template<typename OptimizerType>
LogisticRegression<MatType>::LogisticRegression(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    OptimizerType& optimizer,
    const double lambda) :
    lambda(lambda)
{
  Train(predictors, responses, optimizer);
}

template<typename MatType>
template<typename OptimizerType, typename... CallbackTypes>
double LogisticRegression<MatType>::Train(
        const MatType& predictors,
        const arma::Row<size_t>& responses,
        CallbackTypes&&... callbacks)
{
  OptimizerType optimizer;
  return Train(predictors, responses, optimizer, callbacks...);
}

template<typename MatType>
template<typename OptimizerType, typename... CallbackTypes>
double LogisticRegression<MatType>::Train(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    OptimizerType& optimizer,
    CallbackTypes&&... callbacks)
{
  LogisticRegressionFunction<MatType> errorFunction(predictors, responses,
      lambda);

  // Set size of parameters vector according to the input data received.
  if (parameters.n_elem != predictors.n_rows + 1)
    parameters = arma::rowvec(predictors.n_rows + 1, arma::fill::zeros);

  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(errorFunction, parameters,
      callbacks...);
  Timer::Stop("logistic_regression_optimization");

  Log::Info << "LogisticRegression::LogisticRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;

  return out;
}

template<typename MatType>
template<typename VecType>
size_t LogisticRegression<MatType>::Classify(const VecType& point,
                                             const double decisionBoundary)
    const
{
  return size_t(1.0 / (1.0 + std::exp(-parameters(0) - arma::dot(point,
      parameters.tail_cols(parameters.n_elem - 1)))) +
      (1.0 - decisionBoundary));
}

template<typename MatType>
void LogisticRegression<MatType>::Classify(const MatType& dataset,
                                           arma::Row<size_t>& labels,
                                           const double decisionBoundary) const
{
  // Calculate sigmoid function for each point.  The (1.0 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  labels = arma::conv_to<arma::Row<size_t>>::from((1.0 /
      (1.0 + arma::exp(-parameters(0) -
      parameters.tail_cols(parameters.n_elem - 1) * dataset))) +
      (1.0 - decisionBoundary));
}

template<typename MatType>
void LogisticRegression<MatType>::Classify(const MatType& dataset,
                                           arma::mat& probabilities) const
{
  // Set correct size of output matrix.
  probabilities.set_size(2, dataset.n_cols);

  probabilities.row(1) = 1.0 / (1.0 + arma::exp(-parameters(0) -
      parameters.tail_cols(parameters.n_elem - 1) * dataset));
  probabilities.row(0) = 1.0 - probabilities.row(1);
}

template<typename MatType>
double LogisticRegression<MatType>::ComputeError(
    const MatType& predictors,
    const arma::Row<size_t>& responses) const
{
  // Construct a new error function.
  LogisticRegressionFunction<> newErrorFunction(predictors, responses,
      lambda);

  return newErrorFunction.Evaluate(parameters);
}

template<typename MatType>
double LogisticRegression<MatType>::ComputeAccuracy(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const double decisionBoundary) const
{
  // Predict responses using the current model.
  arma::Row<size_t> tempResponses;
  Classify(predictors, tempResponses, decisionBoundary);

  // Count the number of responses that were correct.
  size_t count = 0;
  for (size_t i = 0; i < responses.n_elem; ++i)
  {
    if (responses(i) == tempResponses(i))
      count++;
  }

  return (double) (count * 100) / responses.n_elem;
}

template<typename MatType>
template<typename Archive>
void LogisticRegression<MatType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & BOOST_SERIALIZATION_NVP(parameters);
  ar & BOOST_SERIALIZATION_NVP(lambda);
}

} // namespace regression
} // namespace mlpack

#endif // MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
