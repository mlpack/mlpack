/**
 * @file linear_regression.hpp
 * @author James Cline
 *
 * Simple least-squares linear regression.
 */
#ifndef __MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define __MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace regression /** Regression methods. */ {

/**
 * A simple linear regresion algorithm using ordinary least squares.
 */
class LinearRegression
{
 public:
  /**
   * Creates the model.
   *
   * @param predictors X, matrix of data points to create B with.
   * @param responses y, the measured data for each point in X
   */
  LinearRegression(arma::mat& predictors, const arma::vec& responses);

  /**
   * Initialize the model from a file.
   *
   * @param filename the name of the file to load the model from.
   */
  LinearRegression(const std::string& filename);

  /**
   * Destructor - no work done.
   */
  ~LinearRegression();

  /**
   * Calculate y_i for each data point in points.
   *
   * @param points the data points to calculate with.
   * @param predictions y, will contain calculated values on completion.
   */
  void Predict(const arma::mat& points, arma::vec& predictions);

  //! Return the parameters (the b vector).
  const arma::vec& Parameters() const { return parameters; }
  //! Modify the parameters (the b vector).
  arma::vec& Parameters() { return parameters; }

 private:
  /**
   * The calculated B.
   * Initialized and filled by constructor to hold the least squares solution.
   */
  arma::vec parameters;
};

}; // namespace linear_regression
}; // namespace mlpack

#endif // __MLPACK_METHODS_LINEAR_REGRESSCLIN_HPP
