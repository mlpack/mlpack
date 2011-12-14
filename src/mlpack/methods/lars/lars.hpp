/**
 * @file lars.hpp
 * @author Nishant Mehta (niche)
 *
 * Definition of the LARS class, which performs Least Angle Regression and the
 * LASSO.
 */
#ifndef __MLPACK_METHODS_LARS_LARS_HPP
#define __MLPACK_METHODS_LARS_LARS_HPP

#include <armadillo>
#include <mlpack/core.hpp>

#define EPS 1e-16

namespace mlpack {
namespace regression {

// beta is the estimator
// yHat is the prediction from the current estimator

/**
 * An implementation of LARS, a stage-wise homotopy-based algorithm for
 * l1 regularized linear regression (LASSO) and l1+l2 regularized linear
 * regression (Elastic Net).
 * Let X be a matrix where each row is a point and each column is a dimension,
 * and let y be a vector of targets.
 * The Elastic Net problem is to solve
 * min_beta ||X beta - y||_2^2 + lambda_1 ||beta||_1 + 0.5 lambda_2 ||beta||_2^2
 * If lambda_1 > 0, lambda_2 = 0, the problem is the LASSO.
 * If lambda_1 > 0, lambda_2 > 0, the problem is the Elastic Net.
 * If lambda_1 = 0, lambda_2 > 0, the problem is Ridge Regression.
 * If lambda_1 = 0, lambda_2 = 0, the problem is unregularized linear
 *     regression.
 *
 * Note: This algorithm is not recommended for use (in terms of efficiency)
 * when lambda_1 = 0.
 *
 * Only minor modifications are necessary to handle the constrained version of
 * the problem:
 *   min_beta ||X beta - y||_2^2 + 0.5 lambda_2 ||beta||_2^2
 *   subject to ||beta||_1 <= tau
 * Although this option currently is not implemented, it will be implemented
 * very soon.
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
   *    solving linear system. If no, compute full Gram matrix at beginning.
   */
  LARS(const bool useCholesky);

  /**
   * Set the parameters to LARS.  lambda2 defaults to 0.
   *
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *    solving linear system. If no, compute full Gram matrix at beginning.
   * @param lambda1 Regularization parameter for l_1-norm penalty
   */
  LARS(const bool useCholesky,
       const double lambda1);

  /**
   * Set the parameters to LARS.
   *
   * @param useCholesky Whether or not to use Cholesky decomposition when
   *    solving linear system. If no, compute full Gram matrix at beginning.
   * @param lambda1 Regularization parameter for l_1-norm penalty
   * @param lambda2 Regularization parameter for l_2-norm penalty
   */
  LARS(const bool useCholesky,
       const double lambda1,
       const double lambda2);

  ~LARS() { }

  /**
   * Set the Gram matrix (done before calling DoLars).
   *
   * @param matGram Matrix to which to set Gram matrix
   */
  void SetGram(const arma::mat& matGram);

  /**
   * Compute Gram matrix. If elastic net, add lambda2 * identity to diagonal.
   *
   * @param matX Data matrix to use for computing Gram matrix
   */
  void ComputeGram(const arma::mat& matX);

  /**
   * Run LARS.
   *
   * @param matX Input data into the algorithm - a matrix where each row is a
   *    point and each column is a dimension
   * @param y A vector of targets
   */
  void DoLARS(const arma::mat& matX, const arma::vec& y);

  /*
   * Load the solution vector, which is the last vector from the solution path
   */
  void Solution(arma::vec& beta);

  //! Accessor for activeSet.
  const std::vector<arma::u32>& ActiveSet() const { return activeSet; }

  //! Accessor for betaPath.
  const std::vector<arma::vec>& BetaPath() const { return betaPath; }

  //! Accessor for lambdaPath.
  const std::vector<double>& LambdaPath() const { return lambdaPath; }

  //! Accessor for matUtriCholFactor.
  const arma::mat& MatUtriCholFactor() const { return matUtriCholFactor; }

private:
  // Gram matrix
  arma::mat matGram;

  // Upper triangular cholesky factor; initially 0x0 arma::matrix.
  arma::mat matUtriCholFactor;

  bool useCholesky;

  bool lasso;
  double lambda1;

  bool elasticNet;
  double lambda2;

  // solution path
  std::vector<arma::vec> betaPath;

  // value of lambda1 for each solution in solution path
  std::vector<double> lambdaPath;

  // number of dimensions in active set
  arma::u32 nActive;

  // active set of dimensions
  std::vector<arma::u32> activeSet;

  // active set membership indicator (for each dimension)
  std::vector<bool> isActive;

  // remove activeVarInd'th element from active set
  void Deactivate(arma::u32 activeVarInd);

  // add dimension varInd to active set
  void Activate(arma::u32 varInd);

  // compute "equiangular" direction in output space
  void ComputeYHatDirection(const arma::mat& matX,
                            const arma::vec& betaDirection,
                            arma::vec& yHatDirection);

  // interpolate to compute last solution vector
  void InterpolateBeta();

  void CholeskyInsert(const arma::vec& newX, const arma::mat& X);

  void CholeskyInsert(const arma::vec& newX, const arma::vec& newGramCol);

  void GivensRotate(const arma::vec& x, arma::vec& rotatedX, arma::mat& G);

  void CholeskyDelete(arma::u32 colToKill);

};

}; // namespace regression
}; // namespace mlpack

#endif
