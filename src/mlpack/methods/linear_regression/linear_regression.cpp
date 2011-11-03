#include "linear_regression.hpp"

namespace mlpack {
namespace linear_regression {

LinearRegression::LinearRegression(arma::mat& predictors,
  const arma::colvec& responses)
{
  size_t n_cols, n_rows;

  n_cols = predictors.n_cols;
  n_rows = predictors.n_rows;

  arma::rowvec ones;
  ones.ones(n_cols);
  predictors.insert_rows(0,ones);
  ++n_rows;

  parameters.set_size(n_cols);

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
  size_t n_cols, n_rows;
  n_cols = points.n_cols;
  n_rows = points.n_rows;

  assert(n_rows == parameters.n_rows - 1);

  predictions.set_size(n_cols);
  predictions += parameters(0);
  for(size_t i = 1; i < n_rows; ++i)
  {
    for(size_t j = 0; j < n_cols; ++j)
    {
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
