/**
 * @file linear_regression.cpp
 * @author James Cline
 *
 * Implementation of simple linear regression.
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

void LinearRegression::Predict(const arma::mat& points, arma::vec& predictions)
{
  // We get the number of columns and rows of the dataset.
  const size_t nCols = points.n_cols;
  const size_t nRows = points.n_rows;

  // We want to be sure we have the correct number of dimensions in the dataset.
  Log::Assert(points.n_rows == parameters.n_rows - 1);

  // Get the predictions, but this ignores the intercept value (parameters[0]).
  predictions = arma::trans(arma::trans(
      parameters.subvec(1, parameters.n_elem - 1)) * points);

  // Now add the intercept.
  predictions += parameters(0);
}
