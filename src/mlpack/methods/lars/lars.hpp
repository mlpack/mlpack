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
// responseshat is the prediction from the current estimator

class LARS {
 private:
  arma::mat data;
  arma::vec responses;

  arma::vec xtResponses;
  arma::mat gram;
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
  LARS(const arma::mat& data,
       const arma::vec& responses,
       const bool useCholesky);

  LARS(const arma::mat& data,
       const arma::vec& responses,
       const bool useCholesky,
       const double lambda1);

  LARS(const arma::mat& data,
       const arma::vec& responses,
       const bool useCholesky,
       const double lambda1,
       const double lambda2);

  ~LARS() { }

  void SetGram(const arma::mat& Gram);

  void ComputeGram();

  void ComputeXty();

  void UpdateX(const std::vector<int>& col_inds, const arma::mat& new_cols);

  void UpdateGram(const std::vector<int>& col_inds);

  void UpdateXty(const std::vector<int>& col_inds);

  void PrintGram();

  void SetY(const arma::vec& y);

  void PrintY();

  const std::vector<arma::u32> active_set();

  const std::vector<arma::vec> beta_path();

  const std::vector<double> lambda_path();

  void SetDesiredLambda(double lambda1);

  void DoLARS();

  void Solution(arma::vec& beta);

  void GetCholFactor(arma::mat& R);

  void Deactivate(arma::u32 active_var_ind);

  void Activate(arma::u32 var_ind);

  void ComputeYHatDirection(const arma::vec& beta_direction,
                            arma::vec& responseshat_direction);

  void InterpolateBeta();

  void CholeskyInsert(const arma::vec& new_x, const arma::mat& X);

  void CholeskyInsert(const arma::vec& new_x, const arma::vec& newGramCol);

  void GivensRotate(const arma::vec& x, arma::vec& rotated_x, arma::mat& G);

  void CholeskyDelete(arma::u32 col_to_kill);
};

}; // namespace lars
}; // namespace mlpack

#include "lars_impl.hpp"
#undef INSIDE_LARS_H

#endif
