/**
 * @file logistic_regression_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of the LogisticRegression class.  This implementation supports
 * L2-regularization.
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
    predictors(predictors),
    responses(responses),
    parameters(arma::zeros<arma::vec>(predictors.n_rows + 1)),
    errorFunction(LogisticRegressionFunction(predictors, responses, lambda)),
    optimizer(OptimizerType<LogisticRegressionFunction>(errorFunction))
{
  // Train the model.
  LearnModel();
}

template<template<typename> class OptimizerType>
LogisticRegression<OptimizerType>::LogisticRegression(
    const arma::mat& predictors,
    const arma::vec& responses,
    const arma::mat& initialPoint,
    const double lambda) :
    predictors(predictors),
    responses(responses),
    parameters(arma::zeros<arma::vec>(predictors.n_rows + 1)),
    errorFunction(LogisticRegressionFunction(predictors, responses)),
    optimizer(OptimizerType<LogisticRegressionFunction>(errorFunction))
{
  // Train the model.
  LearnModel();
}

template<template<typename> class OptimizerType>
LogisticRegression<OptimizerType>::LogisticRegression(
    OptimizerType<LogisticRegressionFunction>& optimizer) :
    predictors(optimizer.Function().Predictors()),
    responses(optimizer.Function().Responses()),
    parameters(optimizer.Function().GetInitialPoint()),
    errorFunction(optimizer.Function()),
    optimizer(optimizer)
{
  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");
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
      errorFunction.Lambda());

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
double LogisticRegression<OptimizerType>::LearnModel()
{
  Timer::Start("logistic_regression_optimization");
  const double out = optimizer.Optimize(parameters);
  Timer::Stop("logistic_regression_optimization");

  return out;
}

}; // namespace regression
}; // namespace mlpack

#endif // __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_REGRESSION_IMPL_HPP
