/**
 * @file logistic_regression_function.cpp
 * @author Sumedh Ghaisas
 *
 * Implementation of hte LogisticRegressionFunction class.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "logistic_regression_function.hpp"

using namespace mlpack;
using namespace mlpack::regression;

LogisticRegressionFunction::LogisticRegressionFunction(
    const arma::mat& predictors,
    const arma::vec& responses,
    const double lambda) :
    predictors(predictors),
    responses(responses),
    lambda(lambda)
{
  initialPoint = arma::zeros<arma::mat>(predictors.n_rows + 1, 1);
}

LogisticRegressionFunction::LogisticRegressionFunction(
    const arma::mat& predictors,
    const arma::vec& responses,
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

  // Calculate vectors of sigmoids.  The intercept term is parameters(0, 0) and
  // does not need to be multiplied by any of the predictors.
  const arma::vec exponents = parameters(0, 0) + predictors.t() *
      parameters.col(0).subvec(1, parameters.n_elem - 1);
  const arma::vec sigmoid = 1.0 / (1.0 + arma::exp(-exponents));

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
 * Evaluate the logistic regression objective function, but with only one point.
 * This is useful for optimizers that use a separable objective function, such
 * as SGD.
 */
double LogisticRegressionFunction::Evaluate(const arma::mat& parameters,
                                            const size_t i) const
{
  // Calculate the regularization term.  We must divide by the number of points,
  // so that sum(Evaluate(parameters, [1:points])) == Evaluate(parameters).
  const double regularization = lambda * (1.0 / (2.0 * predictors.n_cols)) *
      arma::dot(parameters.col(0).subvec(1, parameters.n_elem - 1),
                parameters.col(0).subvec(1, parameters.n_elem - 1));

  // Calculate sigmoid.
  const double exponent = parameters(0, 0) + arma::dot(predictors.col(i),
      parameters.col(0).subvec(1, parameters.n_elem - 1));
  const double sigmoid = 1.0 / (1.0 + std::exp(-exponent));

  if (responses[i] == 1)
    return -log(sigmoid) + regularization;
  else
    return -log(1.0 - sigmoid) + regularization;
}

//! Evaluate the gradient of the logistic regression objective function.
void LogisticRegressionFunction::Gradient(const arma::mat& parameters,
                                          arma::mat& gradient) const
{
  // Regularization term.
  arma::mat regularization;
  regularization = lambda * parameters.col(0).subvec(1, parameters.n_elem - 1);

  const arma::vec sigmoids = 1 / (1 + arma::exp(-parameters(0, 0)
      - predictors.t() * parameters.col(0).subvec(1, parameters.n_elem - 1)));

  gradient.set_size(parameters.n_elem);
  gradient[0] = -arma::accu(responses - sigmoids);
  gradient.col(0).subvec(1, parameters.n_elem - 1) = -predictors * (responses -
      sigmoids) + regularization;
}

/**
 * Evaluate the individual gradients of the logistic regression objective
 * function with respect to individual points.  This is useful for optimizers
 * that use a separable objective function, such as SGD.
 */
void LogisticRegressionFunction::Gradient(const arma::mat& parameters,
                                          const size_t i,
                                          arma::mat& gradient) const
{
  // Calculate the regularization term.
  arma::mat regularization;
  regularization = lambda * parameters.col(0).subvec(1, parameters.n_elem - 1)
      / predictors.n_cols;

  const double sigmoid = 1.0 / (1.0 + std::exp(-parameters(0, 0)
      - arma::dot(predictors.col(i), parameters.col(0).subvec(1,
      parameters.n_elem - 1))));

  gradient.set_size(parameters.n_elem);
  gradient[0] = -(responses[i] - sigmoid);
  gradient.col(0).subvec(1, parameters.n_elem - 1) = -predictors.col(i)
      * (responses[i] - sigmoid) + regularization;
}
