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

LinearRegression::LinearRegression(arma::mat& predictors,
                                   const arma::colvec& responses)
{
  /*
   * We want to calculate the a_i coefficients of:
   * \sum_{i=0}^n (a_i * x_i^i)
   * In order to get the intercept value, we will add a row of ones.
   */

  // We store the number of rows of the predictors.
  // Reminder: Armadillo stores the data transposed from how we think of it,
  //           that is, columns are actually rows (see: column major order).
  size_t nCols = predictors.n_cols;

  // Here we add the row of ones to the predictors.
  arma::rowvec ones;
  ones.ones(nCols);
  predictors.insert_rows(0, ones);

  // We set the parameters to the correct size and initialize them to zero.
  parameters.zeros(nCols);

  // We compute the QR decomposition of the predictors.
  // We transpose the predictors because they are in column major order.
  arma::mat Q, R;
  arma::qr(Q, R, arma::trans(predictors));

  // We compute the parameters, B, like so:
  // R * B = Q^T * responses
  // B = Q^T * responses * R^-1
  arma::solve(parameters, R, arma::trans(Q) * responses);

  // We now remove the row of ones we added so the user's data is unmodified.
  predictors.shed_row(0);
}

LinearRegression::LinearRegression(const std::string& filename)
{
  data::Load(filename, parameters, true);
}

LinearRegression::LinearRegression(const LinearRegression& linearRegression)
{
  parameters = linearRegression.parameters;
}

LinearRegression::~LinearRegression()
{ }

void LinearRegression::Predict(const arma::mat& points, arma::vec& predictions)
{
  // We get the number of columns and rows of the dataset.
  const size_t nCols = points.n_cols;
  const size_t nRows = points.n_rows;

  // We want to be sure we have the correct number of dimensions in the dataset.
  Log::Assert(nRows == parameters.n_rows - 1);
  if (nRows != parameters.n_rows -1)
  {
    Log::Fatal << "The test data must have the same number of columns as the "
        "training file.\n";
  }

  predictions.zeros(nCols);
  // We set all the predictions to the intercept value initially.
  predictions += parameters(0);

  // Now we iterate through the dimensions of the data and parameters.
  for (size_t i = 1; i < nRows + 1; ++i)
  {
    // Now we iterate through each row, or point, of the data.
    for (size_t j = 0; j < nCols; ++j)
    {
      // Increment each prediction value by x_i * a_i, or the next dimensional
      // coefficient and x value.
      predictions(j) += parameters(i) * points(i - 1, j);
    }
  }
}
