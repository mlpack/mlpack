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

  // The number of columns and rows
  size_t n_cols, n_rows;

  n_cols = predictors.n_cols;
  n_rows = predictors.n_rows;

  // Add a row of ones, to get the intercept
  arma::rowvec ones;
  ones.ones(n_cols);
  predictors.insert_rows(0,ones);
  // We have an additional row, now
  ++n_rows;

  // Set the parameters to the correct size, all zeros.
  parameters.zeros(n_cols);

  // inverse( A^T * A ) * A^T * responses, where A = predictors
  parameters = arma::inv((predictors * arma::trans(predictors))) *
    predictors * responses;
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
  for(size_t i = 1; i < n_rows; ++i)
  {
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
