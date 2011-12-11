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
namespace lars {

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
 * If lambda_1 = 0, lambda_2 = 0, the problem is unregularized linear regression.
 * Note: This algorithm is not recommended for use (in terms of efficiency) 
 * when lambda_1 = 0.
 *
 * For more details, see the following papers:
 *
 *
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
 *
 *
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
 */
class LARS {

 public:
  
  LARS(const arma::mat& matX,
       const arma::vec& y,
       const bool useCholesky);

  LARS(const arma::mat& matX,
       const arma::vec& y,
       const bool useCholesky,
       const double lambda1);

  LARS(const arma::mat& matX,
       const arma::vec& y,
       const bool useCholesky,
       const double lambda1,
       const double lambda2);

  ~LARS() { }

  void SetGram(const arma::mat& matGram);

  void ComputeGram();

  void ComputeXty();

  const std::vector<arma::u32> ActiveSet();

  const std::vector<arma::vec> BetaPath();

  const std::vector<double> LambdaPath();

  const arma::mat MatUtriCholFactor();
  
  void DoLARS();

  void Solution(arma::vec& beta);

  
private:
  arma::mat matX;
  arma::vec y;

  arma::vec vecXTy;
  arma::mat matGram;
  
  // Upper triangular cholesky factor; initially 0x0 arma::matrix.
  arma::mat matUtriCholFactor;

  bool useCholesky;

  bool lasso;
  double lambda1;

  bool elasticNet;
  double lambda2;

  std::vector<arma::vec> betaPath;
  std::vector<double> lambdaPath;

  arma::u32 nActive;
  std::vector<arma::u32> activeSet;
  std::vector<bool> isActive;
  
  
  void Deactivate(arma::u32 activeVarInd);

  void Activate(arma::u32 varInd);

  void ComputeYHatDirection(const arma::vec& betaDirection,
                            arma::vec& yHatDirection);

  void InterpolateBeta();

  void CholeskyInsert(const arma::vec& newX, const arma::mat& X);

  void CholeskyInsert(const arma::vec& newX, const arma::vec& newGramCol);

  void GivensRotate(const arma::vec& x, arma::vec& rotatedX, arma::mat& G);

  void CholeskyDelete(arma::u32 colToKill);

  
};

}; // namespace lars
}; // namespace mlpack

#endif
