/**
 * @file linear_regression.cpp
 * @author James Cline
 * @author Michael Fox
 *
 * Implementation of simple linear regression.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::regression;

LinearRegression::LinearRegression(const arma::mat& predictors,
                                   const arma::vec& responses,
                                   const double lambda,
                                   const bool intercept,
                                   const arma::vec& weights) :
    lambda(lambda),
    intercept(intercept)
{
  Train(predictors, responses, intercept, weights);
}

LinearRegression::LinearRegression(const LinearRegression& linearRegression) :
    parameters(linearRegression.parameters),
    lambda(linearRegression.lambda)
{ /* Nothing to do. */ }

void LinearRegression::Train(const arma::mat& predictors,
                             const arma::vec& responses,
                             const bool intercept,
                             const arma::vec& weights)
{
  this->intercept = intercept;

  /*
   * We want to calculate the a_i coefficients of:
   * \sum_{i=0}^n (a_i * x_i^i)
   * In order to get the intercept value, we will add a row of ones.
   */

  // We store the number of rows and columns of the predictors.
  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).
  const size_t nCols = predictors.n_cols;

  arma::mat p = predictors;
  arma::vec r = responses;

  // Here we add the row of ones to the predictors.
  // The intercept is not penalized. Add an "all ones" row to design and set
  // intercept = false to get a penalized intercept.
  if (intercept)
  {
    p.insert_rows(0, arma::ones<arma::mat>(1,nCols));
  }

  if (weights.n_elem > 0)
  {
    p = p * diagmat(sqrt(weights));
    r = sqrt(weights) % responses;
  }

  if (lambda != 0.0)
  {
    // Add the identity matrix to the predictors (this is equivalent to ridge
    // regression).  See http://math.stackexchange.com/questions/299481/ for
    // more information.
    p.insert_cols(nCols, predictors.n_rows);
    p.submat(p.n_rows - predictors.n_rows, nCols, p.n_rows - 1, nCols +
    predictors.n_rows - 1) = sqrt(lambda) *
        arma::eye<arma::mat>(predictors.n_rows, predictors.n_rows);
  }

  // We compute the QR decomposition of the predictors.
  // We transpose the predictors because they are in column major order.
  arma::mat Q, R;
  arma::qr(Q, R, arma::trans(p));

  // We compute the parameters, B, like so:
  // R * B = Q^T * responses
  // B = Q^T * responses * R^-1
  // If lambda > 0, then we must add a bunch of empty responses.
  if (lambda == 0.0)
  {
    arma::solve(parameters, R, arma::trans(Q) * r);
  }
  else
  {
    // Copy responses into larger vector.
    r.insert_rows(nCols,p.n_cols - nCols);
    arma::solve(parameters, R, arma::trans(Q) * r);
  }
}

void LinearRegression::Predict(const arma::mat& points, arma::vec& predictions)
    const
{
  if (intercept)
  {
    // We want to be sure we have the correct number of dimensions in the
    // dataset.
    Log::Assert(points.n_rows == parameters.n_rows - 1);
    // Get the predictions, but this ignores the intercept value
    // (parameters[0]).
    predictions = arma::trans(arma::trans(parameters.subvec(1,
        parameters.n_elem - 1)) * points);
    // Now add the intercept.
    predictions += parameters(0);
  }
  else
  {
    // We want to be sure we have the correct number of dimensions in the dataset.
    Log::Assert(points.n_rows == parameters.n_rows);
    predictions = arma::trans(arma::trans(parameters) * points);
  }

}

//! Compute the L2 squared error on the given predictors and responses.
double LinearRegression::ComputeError(const arma::mat& predictors,
                                      const arma::vec& responses) const
{
  // Get the number of columns and rows of the dataset.
  const size_t nCols = predictors.n_cols;
  const size_t nRows = predictors.n_rows;

  // Calculate the differences between actual responses and predicted responses.
  // We must also add the intercept (parameters(0)) to the predictions.
  arma::vec temp;
  if (intercept)
  {
    // Ensure that we have the correct number of dimensions in the dataset.
    if (nRows != parameters.n_rows - 1)
    {
      Log::Fatal << "The test data must have the same number of columns as the "
          "training file." << std::endl;
    }
    temp = responses - arma::trans( (arma::trans(parameters.subvec(1,
        parameters.n_elem - 1)) * predictors) + parameters(0));
  }
  else
  {
    // Ensure that we have the correct number of dimensions in the dataset.
    if (nRows != parameters.n_rows)
    {
      Log::Fatal << "The test data must have the same number of columns as the "
          "training file." << std::endl;
    }
    temp = responses - arma::trans((arma::trans(parameters) * predictors));
  }
  const double cost = arma::dot(temp, temp) / nCols;

  return cost;
}
