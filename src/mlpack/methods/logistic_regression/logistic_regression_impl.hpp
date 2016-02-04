/**
 * @file logistic_regression_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegression class.  This implementation supports
 * L2-regularization.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
#define __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP

// In case it hasn't been included yet.
#include "logistic_regression.hpp"

namespace mlpack {
namespace regression {

template<typename MatType>
LogisticRegression<MatType>::LogisticRegression(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const double lambda) :
    parameters(arma::zeros<arma::vec>(predictors.n_rows + 1)),
    lambda(lambda)
{
  Train(predictors, responses);
}

template<typename MatType>
LogisticRegression<MatType>::LogisticRegression(
    const MatType& predictors,
    const arma::Row<size_t>& responses,
    const arma::vec& initialPoint,
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
    parameters(dimensionality + 1 /* include intercept term */),
    lambda(lambda)
{
  // No training to do here.
}

template<typename MatType>
template<template<typename> class OptimizerType>
LogisticRegression<MatType>::LogisticRegression(
    OptimizerType<LogisticRegressionFunction<MatType>>& optimizer) :
    parameters(optimizer.Function().GetInitialPoint()),
    lambda(optimizer.Function().Lambda())
{
  Train(optimizer);
}

template<typename MatType>
template<template<typename> class OptimizerType>
void LogisticRegression<MatType>::Train(const MatType& predictors,
                                        const arma::Row<size_t>& responses)
{
  LogisticRegressionFunction<MatType> errorFunction(predictors, responses,
      lambda);
  errorFunction.InitialPoint() = parameters;
  OptimizerType<LogisticRegressionFunction<MatType>> optimizer(errorFunction);

  // Train the model.
  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");

  Log::Info << "LogisticRegression::LogisticRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<typename MatType>
template<template<typename> class OptimizerType>
void LogisticRegression<MatType>::Train(
    OptimizerType<LogisticRegressionFunction<MatType>>& optimizer)
{
  // Everything is good.  Just train the model.
  parameters = optimizer.Function().GetInitialPoint();

  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");

  Log::Info << "LogisticRegression::LogisticRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<typename MatType>
void LogisticRegression<MatType>::Predict(const MatType& predictors,
                                          arma::Row<size_t>& responses,
                                          const double decisionBoundary) const
{
  // Calculate sigmoid function for each point.  The (1.0 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  responses = arma::conv_to<arma::Row<size_t>>::from((1.0 /
      (1.0 + arma::exp(-parameters(0) - predictors.t() *
      parameters.subvec(1, parameters.n_elem - 1)))) +
      (1.0 - decisionBoundary));
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
  Predict(predictors, tempResponses, decisionBoundary);

  // Count the number of responses that were correct.
  size_t count = 0;
  for (size_t i = 0; i < responses.n_elem; i++)
  {
    if (responses(i) == tempResponses(i))
      count++;
  }

  return (double) (count * 100) / responses.n_elem;
}

template<typename MatType>
template<typename Archive>
void LogisticRegression<MatType>::Serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar & data::CreateNVP(parameters, "parameters");
  ar & data::CreateNVP(lambda, "lambda");
}

} // namespace regression
} // namespace mlpack

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
