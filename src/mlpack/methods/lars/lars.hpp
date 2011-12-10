/**
 * @file lars.hpp
 * @author Nishant Mehta (niche)
 *
 * Definition of the LARS class, which performs Least Angle Regression and the
 * LASSO.
 */
#ifndef __MLPACK_METHODS_LARS_LARS_HPP
#define __MLPACK_METHODS_LARS_LARS_HPP

#include <mlpack/core.hpp>

#define EPS 1e-16

namespace mlpack {
namespace lars {

// beta is the estimator
// yHat is the prediction from the current estimator

class LARS {
 private:
  arma::mat matX;
  arma::vec y;

  arma::vec matXTy;
  arma::mat matGram;
  //! Upper triangular cholesky factor; initially 0x0 arma::matrix.
  arma::mat utriCholFactor;

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

  void UpdateX(const std::vector<int>& colInds, const arma::mat& matNewCols);

  void UpdateGram(const std::vector<int>& colInds);

  void UpdateXty(const std::vector<int>& colInds);

  void PrintGram();

  void SetY(const arma::vec& y);

  void PrintY();

  const std::vector<arma::u32> ActiveSet();

  const std::vector<arma::vec> BetaPath();

  const std::vector<double> LambdaPath();

  void SetDesiredLambda(double lambda1);

  void DoLARS();

  void Solution(arma::vec& beta);

  void GetCholFactor(arma::mat& matR);

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

#include "lars_impl.hpp"
#undef INSIDE_LARS_H

#endif
