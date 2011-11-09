#include "linear_regression.hpp"

namespace mlpack {
namespace linear_regression {

LinearRegression::LinearRegression(arma::mat& predictors,
  const arma::colvec& responses)
{

  /*
   * We want to calculate the a_i coefficients of:
   * \sum_{i=0}^n (a_i * x_i^i)
   * We add a row of ones to get a_0, where x_0^0 = 1, the intercept.
   */

  // The number of rows
  size_t n_cols;

  n_cols = predictors.n_cols;

  // Add a row of ones, to get the intercept
  arma::rowvec ones;
  ones.ones(n_cols);
  predictors.insert_rows(0,ones);

  // Set the parameters to the correct size, all zeros.
  parameters.zeros(n_cols);

  // Compute the QR decomposition
  arma::mat Q, R;
  arma::qr(Q,R,arma::trans(predictors));

  // Compute the parameters, R*B=Q^T*responses
  arma::solve( parameters, R, arma::trans(Q)*responses);

  // Remove the added row.
  predictors.shed_row(0);
}

LinearRegression::LinearRegression(const std::string& filename)
{
  parameters.load(filename);
}

LinearRegression::~LinearRegression()
{
}

void LinearRegression::predict(arma::rowvec& predictions, const arma::mat& points)
{
  // The number of columns and rows
  size_t n_cols, n_rows;
  n_cols = points.n_cols;
  n_rows = points.n_rows;

  // Sanity check
  assert(n_rows == parameters.n_rows - 1);

  predictions.zeros(n_cols);
  // Set to a_0
  predictions += parameters(0);

  // Iterate through the dimensions
  for(size_t i = 1; i < n_rows+1; ++i)
  {
    // Iterate through the datapoints
    for(size_t j = 0; j < n_cols; ++j)
    {
      // Add in the next term: a_i * x_i
      predictions(j) += parameters(i) * points(i-1,j);

    }
  }
}

arma::vec LinearRegression::getParameters()
{
  return parameters;
}


bool LinearRegression::load(const std::string& filename)
{
  return data::Load(filename, parameters);
}

bool LinearRegression::save(const std::string& filename)
{
  return data::Save(filename, parameters);
}

}; // namespace linear_regression
}; // namespace mlpack
