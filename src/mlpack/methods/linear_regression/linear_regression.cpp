/**
 * @file linear_regression.cpp
 * @author James Cline
 *
 * Implementation of simple linear regression.
 *
 * This file is part of MLPACK 1.0.6.
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
#include "linear_regression.hpp"

using namespace mlpack;
using namespace mlpack::regression;

LinearRegression::LinearRegression(const arma::mat& predictors,
                                   const arma::colvec& responses,
                                   const double lambda) :
    lambda(lambda)
{
  /*
   * We want to calculate the a_i coefficients of:
   * \sum_{i=0}^n (a_i * x_i^i)
   * In order to get the intercept value, we will add a row of ones.
   */

  // We store the number of rows and columns of the predictors.
  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).
  const size_t nCols = predictors.n_cols;

  // Here we add the row of ones to the predictors.
  arma::mat p;
  if (lambda == 0.0)
  {
    p.set_size(predictors.n_rows + 1, nCols);
    p.submat(1, 0, p.n_rows - 1, nCols - 1) = predictors;
    p.row(0).fill(1);
  }
  else
  {
    // Add the identity matrix to the predictors (this is equivalent to ridge
    // regression).  See http://math.stackexchange.com/questions/299481/ for
    // more information.
    p.set_size(predictors.n_rows + 1, nCols + predictors.n_rows + 1);
    p.submat(1, 0, p.n_rows - 1, nCols - 1) = predictors;
    p.row(0).subvec(0, nCols - 1).fill(1);
    p.submat(0, nCols, p.n_rows - 1, nCols + predictors.n_rows) =
        lambda * arma::eye<arma::mat>(predictors.n_rows + 1,
                                      predictors.n_rows + 1);
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
    arma::solve(parameters, R, arma::trans(Q) * responses);
  }
  else
  {
    // Copy responses into larger vector.
    arma::vec r(nCols + predictors.n_rows + 1);
    r.subvec(0, nCols - 1) = responses;
    r.subvec(nCols, nCols + predictors.n_rows).fill(0);

    arma::solve(parameters, R, arma::trans(Q) * r);
  }
}

LinearRegression::LinearRegression(const std::string& filename) :
    lambda(0.0)
{
  data::Load(filename, parameters, true);
}

LinearRegression::LinearRegression(const LinearRegression& linearRegression) :
    parameters(linearRegression.parameters),
    lambda(linearRegression.lambda)
{ /* Nothing to do. */ }

void LinearRegression::Predict(const arma::mat& points, arma::vec& predictions)
    const
{
  // We want to be sure we have the correct number of dimensions in the dataset.
  Log::Assert(points.n_rows == parameters.n_rows - 1);

  // Get the predictions, but this ignores the intercept value (parameters[0]).
  predictions = arma::trans(arma::trans(
      parameters.subvec(1, parameters.n_elem - 1)) * points);

  // Now add the intercept.
  predictions += parameters(0);
}

//! Compute the L2 squared error on the given predictors and responses.
double LinearRegression::ComputeError(const arma::mat& predictors,
                                      const arma::vec& responses) const
{
  // Get the number of columns and rows of the dataset.
  const size_t nCols = predictors.n_cols;
  const size_t nRows = predictors.n_rows;

  // Ensure that we have the correct number of dimensions in the dataset.
  if (nRows != parameters.n_rows - 1)
  {
    Log::Fatal << "The test data must have the same number of columns as the "
        "training file." << std::endl;
  }

  // Calculate the differences between actual responses and predicted responses.
  // We must also add the intercept (parameters(0)) to the predictions.
  arma::vec temp = responses - arma::trans(
      (arma::trans(parameters.subvec(1, parameters.n_elem - 1)) * predictors) +
      parameters(0));

  const double cost = arma::dot(temp, temp) / nCols;

  return cost;
}
