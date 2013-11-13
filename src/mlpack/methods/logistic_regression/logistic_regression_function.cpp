/**
 * @file logistic_regression_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of hte LogisticRegressionFunction class.
 */
#include "logistic_regression_function.hpp"

using namespace mlpack;
using namespace mlpack::regression;

LogisticRegressionFunction::LogisticRegressionFunction(
    arma::mat& predictors,
    arma::vec& responses,
    const double lambda) :
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  initialPoint = arma::zeros<arma::mat>(predictors.n_rows + 1, 1);
}

LogisticRegressionFunction::LogisticRegressionFunction(
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
  if (initialPoint.n_rows != (predictors.n_rows + 1) ||
      initialPoint.n_cols != 1)
    this->initialPoint = arma::zeros<arma::mat>(predictors.n_rows + 1, 1);
}

/*
arma::vec LogisticRegressionFunction::getSigmoid(const arma::vec& values,
                                       arma::vec& output) const
{
  arma::vec out = arma::ones<arma::vec>(values.n_rows,1) /
      (arma::ones<arma::vec>(values.n_rows,1) + arma::exp(-values));
  return out;
}
*/

/**
 * Evaluate the logistic regression objective function given the estimated
 * parameters.
 */
double LogisticRegressionFunction::Evaluate(const arma::mat& parameters)
    const
{
  // The objective function is the log-likelihood function (w is the parameters
  // vector for the model; y is the responses; x is the predictors; sig() is the
  // sigmoid function):
  //   f(w) = sum(y log(sig(w'x)) + (1 - y) log(sig(1 - w'x))).
  // We want to minimize this function.  L2-regularization is just lambda
  // multiplied by the squared l2-norm of the parameters then divided by two.

  // For the regularization, we ignore the first term, which is the intercept
  // term.
  const double regularization = 0.5 * lambda *
      arma::dot(parameters.col(0).subvec(1, parameters.n_elem - 1),
                parameters.col(0).subvec(1, parameters.n_elem - 1));

  // Calculate vectors of sigmoids.
  const arma::vec exponents = predictors.t() * parameters;
  const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-exponents));

  // Assemble full objective function.  Often the objective function and the
  // regularization as given are divided by the number of features, but this
  // doesn't actually affect the optimization result, so we'll just ignore those
  // terms for computational efficiency.
  double result = 0.0;
  for (size_t i = 0; i < responses.n_elem; ++i)
  {
    if (responses[i] == 1)
      result += responses[i] * log(sigmoid[i]);
    else
      result += (1 - responses[i]) * log(1.0 - sigmoid[i]);
  }

  // Invert the result, because it's a minimization.
  return -(result + regularization);
}

void LogisticRegressionFunction::Gradient(const arma::mat& parameters,
                                          arma::mat& gradient)
{
  // Regularization term.
  arma::mat regularization = arma::zeros<arma::mat>(predictors.n_rows, 1);
  regularization.rows(1, predictors.n_rows - 1) = lambda *
      parameters.col(0).subvec(1, predictors.n_rows - 1);

  gradient = -predictors * (responses
      - (1 / (1 + arma::exp(-predictors.t() * parameters))))
      - regularization;
}
