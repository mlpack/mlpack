/** @file lars.h
 *
 *  This file implements Least Angle Regression
 *
 *  @author Nishant Mehta (niche)
 *  @bug No known bugs.
 */

// beta is the estimator
// mu is the prediction from the current estimator

// notes: we currently do not require the entire regularization path, so we just keep track of the previous beta and the current beta


#ifndef LARS_H
#define LARS_H

#define EPS 1e-13

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

  std::vector<vec> beta_path_;
  std::vector<double> lambda_path_;
  
  
 public:
  Lars() { }

  ~Lars() { }

  void Init(mat &X, vec &y) {
    X_ = mat(X);
    y_ = vec(y);
    
    n_ = X_.n_rows;
    p_ = X_.n_cols;

    ComputeGram();
    ComputeXty();
  }

  
  void ComputeGram() {
    Gram_ = trans(X_) * X_;
  }
  
  
  void ComputeXty() {
    Xty_ = trans(X_) * y_;
  }    
  
  
  void UpdateX(std::vector<int> &col_inds, mat &new_cols) {
    for(u32 i = 0; i < col_inds.size(); i++) {
      X_.col(col_inds[i]) = new_cols.col(i);
    }
    
    UpdateGram(col_inds);
    UpdateXty(col_inds);
  }

  
  void UpdateGram(std::vector<int> &col_inds) {
    for (std::vector<int>::iterator i = col_inds.begin(); 
	 i != col_inds.end(); 
	 ++i) {
      for (std::vector<int>::iterator j = col_inds.begin(); 
	   j != col_inds.end(); 
	   ++j) {
	Gram_(*i, *j) = dot(X_.col(*i), X_.col(*j));
      }
    }
  }
  
  void UpdateXty(std::vector<int> &col_inds) {
    for (std::vector<int>::iterator i = col_inds.begin(); 
	 i != col_inds.end(); 
	 ++i) {
      Xty_(*i) = dot(X_.col(*i), y_);
    }
  }
  
  
  void PrintGram() {
    Gram_.print("Gram matrix");
  }
  
  
  void SetY(vec &y) {
    y_ = y; // I don't know how to copy the values of vectors yet, so this will have to do. This is wasteful though because we reallocate memory for y_
  }

  
  void PrintY() {
    y_.print();
  }

  
  const std::vector<vec> beta_path() {
    return beta_path_;
  }

  
  const std::vector<double> lambda_path() {
    return lambda_path_;
  }
  
  
  // vanilla LARS - using Gram matrix
  void DoLARS() {
    std::vector<int> active_set(0);
    std::vector<bool> inactive_set(p_);
    fill(inactive_set.begin(), inactive_set.end(), true);
    
    // initialize mu and beta
    vec beta = zeros(p_);
    vec mu = zeros(n_);
    
    u32 n_active = 0;
    
    vec corr = trans(X_) * y_;
    vec abs_corr = abs(corr);
    u32 best_ind;
    double max_corr = abs_corr.max(best_ind);
    
    //std::vector<vec> mu_path;
    
    beta_path_.push_back(beta);
    //mu_path.push_back(mu);
    lambda_path_.push_back(max_corr);
    
    
    // MAIN LOOP
    while((n_active < p_) && (max_corr > EPS)) {
      //printf("n_active = %d\n", n_active);
      n_active++;
      //printf("best_ind = %d\n", best_ind);
      active_set.push_back(best_ind);
      inactive_set[best_ind] = 0;

      
      // compute lambda
      //double lambda = 0;
      //for(u32 i = 0; i < n_active; i++) {
      //lambda += fabs(dot(Gram_.col(active_set[i]), beta) - Xty_(active_set[i]));
      //}
      //lambda /= ((double) n_active);
      //lambda_path.push_back(lambda);
      
      
      // compute signs of correlations
      vec s = vec(n_active);
      for(u32 i = 0; i < n_active; i++) {
	s(i) = corr(active_set[i]) / fabs(corr(active_set[i]));
      }
      mat S = s * ones<mat>(1, n_active);
      
      mat Gram_active = mat(n_active, n_active);
      for(u32 i = 0; i < n_active; i++) {
	for(u32 j = 0; j < n_active; j++) {
	  Gram_active(i,j) = Gram_(active_set[i], active_set[j]);
	}
      }

      // compute "equiangular" direction in parameter space
      vec unnormalized_beta_direction = inv(Gram_active % trans(S) % S) * ones<mat>(n_active, 1);
      double normalization = 1.0 / sqrt(sum(unnormalized_beta_direction));
      vec beta_direction = unnormalized_beta_direction * normalization;
      
      // compute "equiangular" direction in output space
      vec mu_direction = zeros(n_);
      for(u32 i = 0; i < n_active; i++) {
	mu_direction += beta_direction(i) * s(i) * X_.col(active_set[i]);
      }

      // if not all variables are active
      double gamma = max_corr / normalization;
      best_ind = -1;
      if(n_active < p_) {
	// compute correlations with direction
	for(u32 ind = 0; ind < p_; ind++) {
	  if(!inactive_set[ind]) {
	    continue;
	  }
	  double dir_corr = dot(X_.col(ind), mu_direction);
	  double val1 = (max_corr - corr(ind)) / (normalization - dir_corr);
	  double val2 = (max_corr + corr(ind)) / (normalization + dir_corr);
	  if(val1 > 0) {
	    if(val1 < gamma) {
	      gamma = val1;
	      best_ind = ind;
	    }
	  }
	  if(val2 > 0) {
	    if(val2 < gamma) {
	      gamma = val2;
	      best_ind = ind;
	    }
	  }
	}
      }

      // update prediction
      mu += gamma * mu_direction;
      
      // update estimator
      for(u32 i = 0; i < n_active; i++) {
	beta(active_set[i]) += gamma * beta_direction(i);
      }
      beta_path_.push_back(beta);
      //mu_path.push_back(mu);
      
      // compute correlates
      corr = trans(X_) * (y_ - mu);
      max_corr -= gamma * normalization;
      lambda_path_.push_back(max_corr);
    }
    
    /*
    for(u32 i = 0; i < active_set.size(); i++) {
      printf("a %d\n", active_set[i]);
    }
    */
  }

};

#endif
