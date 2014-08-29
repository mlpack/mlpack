/**
 * @file lars.hpp
 * @author Nishant Mehta (niche)
 *
 * Definition of the LARS class, which performs Least Angle Regression and the
 * LASSO.
 *
 * Only minor modifications of LARS are necessary to handle the constrained
 * version of the problem:
 *
 * \f[
 * \min_{\beta} 0.5 || X \beta - y ||_2^2 + 0.5 \lambda_2 || \beta ||_2^2
 * \f]
 * subject to \f$ ||\beta||_1 <= \tau \f$
 *
 * Although this option currently is not implemented, it will be implemented
 * very soon.
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_METHODS_LARS_LARS_HPP
#define __MLPACK_METHODS_LARS_LARS_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace regression {

// beta is the estimator
// yHat is the prediction from the current estimator

/**
 * An implementation of LARS, a stage-wise homotopy-based algorithm for
 * l1-regularized linear regression (LASSO) and l1+l2 regularized linear
 * regression (Elastic Net).
 *
 * Let \f$ X \f$ be a matrix where each row is a point and each column is a
 * dimension and let \f$ y \f$ be a vector of responses.
 *
 * The Elastic Net problem is to solve
 *
 * \f[ \min_{\beta} 0.5 || X \beta - y ||_2^2 + \lambda_1 || \beta ||_1 +
 *     0.5 \lambda_2 || \beta ||_2^2 \f]
 *
 * where \f$ \beta \f$ is the vector of regression coefficients.
 *
 * If \f$ \lambda_1 > 0 \f$ and \f$ \lambda_2 = 0 \f$, the problem is the LASSO.
 * If \f$ \lambda_1 > 0 \f$ and \f$ \lambda_2 > 0 \f$, the problem is the
 *   elastic net.
 * If \f$ \lambda_1 = 0 \f$ and \f$ \lambda_2 > 0 \f$, the problem is ridge
 *   regression.
 * If \f$ \lambda_1 = 0 \f$ and \f$ \lambda_2 = 0 \f$, the problem is
 *   unregularized linear regression.
 *
 * Note: This algorithm is not recommended for use (in terms of efficiency)
 * when \f$ \lambda_1 \f$ = 0.
 *
 * For more details, see the following papers:
 *
 * @code
 * @article{efron2004least,
 *   title={Least angle regression},
 *   author={Efron, B. and Hastie, T. and Johnstone, I. and Tibshirani, R.},
 *   journal={The Annals of statistics},
 *   volume={32},
 *   number={2},
 *   pages={407--499},
 *   year={2004},
 *   publisher={Institute of Mathematical Statistics}
 * }
 * @endcode
 *
 * @code
 * @article{zou2005regularization,
 *   title={Regularization and variable selection via the elastic net},
 *   author={Zou, H. and Hastie, T.},
 *   journal={Journal of the Royal Statistical Society Series B},
 *   volume={67},
 *   number={2},
 *   pages={301--320},
 *   year={2005},
 *   publisher={Royal Statistical Society}
 * }
 * @endcode
 */
class LARS
{
 public:
  /**
   * Set the parameters to LARS.  Both lambda1 and lambda2 default to 0.
   *
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *    solving linear system (as opposed to using the full Gram matrix).
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param tolerance Run until the maximum correlation of elements in (X^T y)
   *     is less than this.
   */
  LARS(const bool useCholesky,
       const double lambda1 = 0.0,
       const double lambda2 = 0.0,
       const double tolerance = 1e-16);

  /**
   * Set the parameters to LARS, and pass in a precalculated Gram matrix.  Both
   * lambda1 and lambda2 default to 0.
   *
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *    solving linear system (as opposed to using the full Gram matrix).
   * @param gramMatrix Gram matrix.
   * @param lambda1 Regularization parameter for l1-norm penalty.
   * @param lambda2 Regularization parameter for l2-norm penalty.
   * @param tolerance Run until the maximum correlation of elements in (X^T y)
   *     is less than this.
   */
  LARS(const bool useCholesky,
       const arma::mat& gramMatrix,
       const double lambda1 = 0.0,
       const double lambda2 = 0.0,
       const double tolerance = 1e-16);

  /**
   * Run LARS.  The input matrix (like all MLPACK matrices) should be
   * column-major -- each column is an observation and each row is a dimension.
   * However, because LARS is more efficient on a row-major matrix, this method
   * will (internally) transpose the matrix.  If this transposition is not
   * necessary (i.e., you want to pass in a row-major matrix), pass 'false' for
   * the transposeData parameter.
   *
   * @param data Column-major input data (or row-major input data if rowMajor =
   *     true).
   * @param responses A vector of targets.
   * @param beta Vector to store the solution (the coefficients) in.
   * @param rowMajor Set to false if the data is row-major.
   */
  void Regress(const arma::mat& data,
               const arma::vec& responses,
               arma::vec& beta,
               const bool transposeData = true);

  //! Access the set of active dimensions.
  const std::vector<size_t>& ActiveSet() const { return activeSet; }

  //! Access the set of coefficients after each iteration; the solution is the
  //! last element.
  const std::vector<arma::vec>& BetaPath() const { return betaPath; }

  //! Access the set of values for lambda1 after each iteration; the solution is
  //! the last element.
  const std::vector<double>& LambdaPath() const { return lambdaPath; }

  //! Access the upper triangular cholesky factor
  const arma::mat& MatUtriCholFactor() const { return matUtriCholFactor; }

  // Returns a string representation of this object.
  std::string ToString() const;

 private:
  //! Gram matrix.
  arma::mat matGramInternal;

  //! Reference to the Gram matrix we will use.
  const arma::mat& matGram;

  //! Upper triangular cholesky factor; initially 0x0 matrix.
  arma::mat matUtriCholFactor;

  //! Whether or not to use Cholesky decomposition when solving linear system.
  bool useCholesky;

  //! True if this is the LASSO problem.
  bool lasso;
  //! Regularization parameter for l1 penalty.
  double lambda1;

  //! True if this is the elastic net problem.
  bool elasticNet;
  //! Regularization parameter for l2 penalty.
  double lambda2;

  //! Tolerance for main loop.
  double tolerance;

  //! Solution path.
  std::vector<arma::vec> betaPath;

  //! Value of lambda_1 for each solution in solution path.
  std::vector<double> lambdaPath;

  //! Active set of dimensions.
  std::vector<size_t> activeSet;

  //! Active set membership indicator (for each dimension).
  std::vector<bool> isActive;

  // Set of variables that are ignored (if any).

  //! Set of ignored variables (for dimensions in span{active set dimensions}).
  std::vector<size_t> ignoreSet;

  //! Membership indicator for set of ignored variables.
  std::vector<bool> isIgnored;

  /**
   * Remove activeVarInd'th element from active set.
   *
   * @param activeVarInd Index of element to remove from active set.
   */
  void Deactivate(const size_t activeVarInd);

  /**
   * Add dimension varInd to active set.
   *
   * @param varInd Dimension to add to active set.
   */
  void Activate(const size_t varInd);

  /**
   * Add dimension varInd to ignores set (never removed).
   *
   * @param varInd Dimension to add to ignores set.
   */
  void Ignore(const size_t varInd);

  // compute "equiangular" direction in output space
  void ComputeYHatDirection(const arma::mat& matX,
                            const arma::vec& betaDirection,
                            arma::vec& yHatDirection);

  // interpolate to compute last solution vector
  void InterpolateBeta();

  void CholeskyInsert(const arma::vec& newX, const arma::mat& X);

  void CholeskyInsert(double sqNormNewX, const arma::vec& newGramCol);

  void GivensRotate(const arma::vec::fixed<2>& x,
                    arma::vec::fixed<2>& rotatedX,
                    arma::mat& G);

  void CholeskyDelete(const size_t colToKill);
};

}; // namespace regression
}; // namespace mlpack

#endif
