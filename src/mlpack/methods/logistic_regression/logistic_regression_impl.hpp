/**
 * @file logistic_regression_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegression class.  This implementation supports
 * L2-regularization.
 *
 * This file is part of mlpack 1.0.12.
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

template<template<typename> class OptimizerType>
LogisticRegression<OptimizerType>::LogisticRegression(
    const arma::mat& predictors,
    const arma::vec& responses,
    const double lambda) :
    parameters(arma::zeros<arma::vec>(predictors.n_rows + 1)),
    lambda(lambda)
{
  LogisticRegressionFunction errorFunction(predictors, responses, lambda);
  OptimizerType<LogisticRegressionFunction> optimizer(errorFunction);

  // Train the model.
  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");

  Log::Info << "LogisticRegression::LogisticRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<template<typename> class OptimizerType>
LogisticRegression<OptimizerType>::LogisticRegression(
    const arma::mat& predictors,
    const arma::vec& responses,
    const arma::mat& initialPoint,
    const double lambda) :
    parameters(arma::zeros<arma::vec>(predictors.n_rows + 1)),
    lambda(lambda)
{
  LogisticRegressionFunction errorFunction(predictors, responses, lambda);
  errorFunction.InitialPoint() = initialPoint;
  OptimizerType<LogisticRegressionFunction> optimizer(errorFunction);

  // Train the model.
  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");

  Log::Info << "LogisticRegression::LogisticRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<template<typename> class OptimizerType>
LogisticRegression<OptimizerType>::LogisticRegression(
    OptimizerType<LogisticRegressionFunction>& optimizer) :
    parameters(optimizer.Function().GetInitialPoint()),
    lambda(optimizer.Function().Lambda())
{
  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");

  Log::Info << "LogisticRegression::LogisticRegression(): final objective of "
      << "trained model is " << out << "." << std::endl;
}

template<template<typename> class OptimizerType>
LogisticRegression<OptimizerType>::LogisticRegression(
    const arma::vec& parameters,
    const double lambda) :
    parameters(parameters),
    lambda(lambda)
{
  // Nothing to do.
}

template<template<typename> class OptimizerType>
void LogisticRegression<OptimizerType>::Predict(const arma::mat& predictors,
                                                arma::vec& responses,
                                                const double decisionBoundary)
    const
{
  // Calculate sigmoid function for each point.  The (1.0 - decisionBoundary)
  // term correctly sets an offset so that floor() returns 0 or 1 correctly.
  responses = arma::floor((1.0 / (1.0 + arma::exp(-parameters(0)
      - predictors.t() * parameters.subvec(1, parameters.n_elem - 1))))
      + (1.0 - decisionBoundary));
}

template <template<typename> class OptimizerType>
double LogisticRegression<OptimizerType>::ComputeError(
    const arma::mat& predictors,
    const arma::vec& responses) const
{
  // Construct a new error function.
  LogisticRegressionFunction newErrorFunction(predictors, responses,
      lambda);

  return newErrorFunction.Evaluate(parameters);
}

template <template<typename> class OptimizerType>
double LogisticRegression<OptimizerType>::ComputeAccuracy(
    const arma::mat& predictors,
    const arma::vec& responses,
    const double decisionBoundary) const
{
  // Predict responses using the current model.
  arma::vec tempResponses;
  Predict(predictors, tempResponses, decisionBoundary);

  // Count the number of responses that were correct.
  size_t count = 0;
  for (size_t i = 0; i < responses.n_elem; i++)
    if (responses(i) == tempResponses(i))
      count++;

  return (double) (count * 100) / responses.n_rows;
}

template <template<typename> class OptimizerType>
std::string LogisticRegression<OptimizerType>::ToString() const
{
  std::ostringstream convert;
  convert << "Logistic Regression [" << this << "]" << std::endl;
  convert << "  Parameters: " << parameters.n_rows << std::endl;
  convert << "  Lambda: " << lambda << std::endl;
  return convert.str();
}

}; // namespace regression
}; // namespace mlpack

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
