/**
 * @file logistic_regression_function_impl.hpp
 * @author Sumedh Ghaisas
 *
 * Implementation of hte LogisticFunction class.
 */
#ifndef __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_FUNCTION_IMPL_HPP
#define __MLPACK_METHODS_LOGISTIC_REGRESSION_LOGISTIC_FUNCTION_IMPL_HPP

// In case it hasn't been done yet.
#include "logistic_regression_function.hpp"

namespace mlpack {
namespace regression {

LogisticFunction::LogisticFunction(
    arma::mat& predictors,
    arma::vec& responses,
    const double lambda) :
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  initialPoint = arma::zeros<arma::mat>(predictors.n_rows + 1, 1);
}

LogisticFunction::LogisticFunction(
    arma::mat& predictors,
    arma::vec& responses,
    const arma::mat& initialPoint,
    const double lambda) :
    initialPoint(initialPoint),
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  //to check if initialPoint is compatible with predictors
  if(initialPoint.n_rows != (predictors.n_rows + 1) || initialPoint.n_cols != 1)
    this->initialPoint = arma::zeros<arma::mat>(predictors.n_rows + 1,1);
}

arma::vec LogisticFunction::getSigmoid(const arma::vec& values) const
{
  arma::vec out = arma::ones<arma::vec>(values.n_rows,1) /
      (arma::ones<arma::vec>(values.n_rows,1) + arma::exp(-values));
  return out;
}

double LogisticFunction::Evaluate(
    const arma::mat& predictors,
    const arma::vec& responses,
    const arma::mat& values) const
{
  size_t nCols = predictors.n_cols;

  //sigmoid = Sigmoid(X' * values)
  arma::vec sigmoid = getSigmoid(arma::trans(predictors) * values);

  //l2-regularization(considering only values(2:end) in regularization
  arma::vec temp = arma::trans(values) * values;
  double regularization = lambda * (temp(0,0) - values(0,0) * values(0,0)) /
      (2 * responses.n_rows);

  //J = -(sum(y' * log(sigmoid)) + sum((ones(m,1) - y)' * log(ones(m,1)
  //   - sigmoid))) + regularization
  return -(sum(arma::trans(responses) * arma::log(sigmoid)) +
      sum(arma::trans(arma::ones<arma::vec>(nCols, 1) - responses) *
      arma::log(arma::ones<arma::vec>(nCols,1) - sigmoid))) /
      predictors.n_cols + regularization;
}

void LogisticFunction::Gradient(
    const arma::mat& values,
    arma::mat& gradient)
{
  //regularization
  arma::mat regularization = arma::zeros<arma::mat>(predictors.n_rows, 1);
  regularization.rows(1, predictors.n_rows - 1) = lambda *
      values.rows(1, predictors.n_rows - 1) / responses.n_rows;

  //gradient =
  gradient = -(predictors * (responses - getSigmoid(arma::trans(predictors) *
      values))) / responses.n_rows + regularization;
}

}; // namespace regression
}; // namespace mlpack

#endif
