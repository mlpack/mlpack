/** @file lars.h
 *
 *  This file implements Least Angle Regression and the LASSO
 *
 *  @author Nishant Mehta (niche)
 *  @bug Possible issue. Search for "I have no idea why..." in lars_impl.h
 */

// beta is the estimator
// y_hat is the prediction from the current estimator

#ifndef LARS_H
#define LARS_H

#define INSIDE_LARS_H


#define EPS 1e-16

using namespace arma;
using namespace std;


class Lars {
 private:
  mat X_;
  vec y_;

  u32 n_;
  u32 p_;
  
  mat Gram_;
  vec Xty_;
  
  bool use_cholesky_;
  
  bool lasso_;
  double lambda_1_;
  
  bool elastic_net_;
  double lambda_2_;
  
  std::vector<vec> beta_path_;
  std::vector<double> lambda_path_;
  
  u32 n_active_;
  std::vector<u32> active_set_;
  std::vector<bool> is_active_;
  
  
 public:
  Lars();
  ~Lars() { }
  
  
  void Init(const mat& X, const vec& y,
	    bool use_cholesky, double lambda_1, double lambda_2);
  
  void Init(const mat& X, const vec& y,
	    bool use_cholesky, double lambda_1);
  
  void Init(const mat& X, const vec& y, bool use_cholesky);
  
  void ComputeGram();
  
  void ComputeXty();
  
  void UpdateX(const std::vector<int>& col_inds, const mat& new_cols);
  
  void UpdateGram(const std::vector<int>& col_inds);
  
  void UpdateXty(const std::vector<int>& col_inds);
  
  void PrintGram();
  
  void SetY(const vec& y);
  
  void PrintY();

  const std::vector<u32> active_set();
  
  const std::vector<vec> beta_path();
  
  const std::vector<double> lambda_path();
  
  void SetDesiredLambda(double lambda_1);
  
  void DoLARS();
  
  void Solution(vec& beta);
  
  void Deactivate(u32 active_var_ind);

  void Activate(u32 var_ind);
  
  void ComputeYHatDirection(const vec& beta_direction,
			    vec& y_hat_direction);
  
  void InterpolateBeta(double ultimate_lambda);
  
  void CholeskyInsert(mat& R, const vec& new_x, const mat& X);
  
  void CholeskyInsert(mat& R, const vec& new_x, const vec& new_Gram_col);
  
  void GivensRotate(const vec& x, vec& rotated_x, mat& G);
  
  void CholeskyDelete(mat& R, u32 col_to_kill);
};

#include "lars_impl.h"
#undef INSIDE_LARS_H

#endif
