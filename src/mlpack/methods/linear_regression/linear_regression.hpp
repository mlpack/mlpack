/**
 * @file linear_regression.hpp
 * @author James Cline
 *
 * Simple least-squares linear regression.
 * This file is part of MLPACK 1.0.2.
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
#ifndef __MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP
#define __MLPACK_METHODS_LINEAR_REGRESSION_LINEAR_REGRESSION_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace regression /** Regression methods. */ {

/**
 * A simple linear regression algorithm using ordinary least squares.
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
   * Copy constructor.
   *
   * @param linearRegression the other instance to copy parameters from.
   */
  LinearRegression(const LinearRegression& linearRegression);

  /**
   * Default constructor.
   */
  LinearRegression() {}


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
