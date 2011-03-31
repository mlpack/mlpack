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

  bool lasso_;
  bool use_cholesky_;

  std::vector<vec> beta_path_;
  std::vector<double> lambda_path_;
  
  double desired_lambda_;

  u32 n_active_;
  std::vector<u32> active_set_;
  std::vector<bool> is_active_;
  
  
  
 public:
  Lars() { }

  ~Lars() { }
  
  void Init(const mat& X, const vec& y, bool lasso, bool use_cholesky) {
    X_ = mat(X);
    y_ = vec(y);
    
    n_ = X_.n_rows;
    p_ = X_.n_cols;
    
    lasso_ = lasso;
    use_cholesky_ = use_cholesky;
    
    ComputeXty();
    if(!use_cholesky_) {
      ComputeGram();
    }
    
    // set up active set variables
    n_active_ = 0;
    active_set_ = std::vector<u32>(0);
    is_active_ = std::vector<bool>(p_);
    fill(is_active_.begin(), is_active_.end(), false);
  }

  
  void ComputeGram() {
    Gram_ = trans(X_) * X_;
  }
  
  
  void ComputeXty() {
    Xty_ = trans(X_) * y_;
  }    
  
  
  void UpdateX(const std::vector<int>& col_inds, const mat& new_cols) {
    for(u32 i = 0; i < col_inds.size(); i++) {
      X_.col(col_inds[i]) = new_cols.col(i);
    }

    if(!use_cholesky_) {
      UpdateGram(col_inds);
    }
    UpdateXty(col_inds);
  }
  
  
  void UpdateGram(const std::vector<int>& col_inds) {
    for (std::vector<int>::const_iterator i = col_inds.begin(); 
	 i != col_inds.end(); 
	 ++i) {
      for (std::vector<int>::const_iterator j = col_inds.begin(); 
	   j != col_inds.end(); 
	   ++j) {
	Gram_(*i, *j) = dot(X_.col(*i), X_.col(*j));
      }
    }
  }
  
  
  void UpdateXty(const std::vector<int>& col_inds) {
    for (std::vector<int>::const_iterator i = col_inds.begin(); 
	 i != col_inds.end(); 
	 ++i) {
      Xty_(*i) = dot(X_.col(*i), y_);
    }
  }
  
  
  
  void PrintGram() {
    Gram_.print("Gram matrix");
  }
  
  
  void SetY(const vec& y) {
    y_ = y;
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
  
  
  void DoLARS(double desired_lambda) {
    SetDesiredLambda(desired_lambda);
    DoLARS();
  }

  
  void SetDesiredLambda(double desired_lambda) {
    desired_lambda_ = desired_lambda;
  }
  
  
  // vanilla LARS - using Gram matrix
  void DoLARS() {
    
    // initialize mu and beta
    vec beta = zeros(p_);
    vec mu = zeros(n_);
    
    //u32 n_active = 0;
    
    bool kick_out = false;
    
    vec corr = Xty_;
    vec abs_corr = abs(corr);
    u32 change_ind;
    double max_corr = abs_corr.max(change_ind); // change_ind gets set here
    
    beta_path_.push_back(beta);
    lambda_path_.push_back(max_corr);
  
    
    mat R; // upper triangular cholesky factor, initially 0 by 0 matrix
    
    // MAIN LOOP
    while((n_active_ < p_) && (max_corr > EPS)) {
      if(kick_out) {
	// index is in position change_ind in active_set
	printf("kick out!\n");
	kick_out = false;
	
	if(use_cholesky_) {
	  CholeskyDelete(R, change_ind);
	}
	
	// remove variable from active set
	//Deactive(change_ind);
	n_active_--;
	is_active_[active_set_[change_ind]] = 0;
	active_set_.erase(active_set_.begin() + change_ind);
      }
      else {
	// index is absolute index
	printf("active!\n");
	
	if(use_cholesky_) {
	  vec new_Gram_col = vec(n_active_);
	  for(u32 i = 0; i < n_active_; i++) {
	    new_Gram_col[i] = 
	      dot(X_.col(active_set_[i]), X_.col(change_ind));
	  }
	  CholeskyInsert(R, X_.col(change_ind), new_Gram_col);
	}
	
	// add variable to active set
	n_active_++;
	is_active_[change_ind] = true;
	active_set_.push_back(change_ind);
      }
      
      
      // compute signs of correlations
      vec s = vec(n_active_);
      for(u32 i = 0; i < n_active_; i++) {
	s(i) = corr(active_set_[i]) / fabs(corr(active_set_[i]));
      }
      mat S = s * ones<mat>(1, n_active_);
      
      
      // compute "equiangular" direction in parameter space (beta_direction)
      /* We use quotes because in the case of non-unit norm variables,
	 this need not be equiangular. */
      vec unnormalized_beta_direction; 
      double normalization;
      vec beta_direction;
      if(use_cholesky_) {
	/* Note that:
	     R^T R % S^T % S = (R % S)^T (R % S)
	   Now, for 1 the ones vector:
	     inv( (R % S)^T (R % S) ) 1
	      = inv(R % S) inv((R % S)^T) 1
	      = inv(R % S) Solve((R % S)^T, 1)
	      = inv(R % S) Solve(R^T, s)
	      = Solve(R % S, Solve(R^T, s)
	      = s % Solve(R, Solve(R^T, s))
	*/
	unnormalized_beta_direction = solve(R, solve(trans(R), s));
	normalization = 1.0 / sqrt(dot(s, unnormalized_beta_direction));
	beta_direction = normalization * unnormalized_beta_direction;
      }
      else{
	mat Gram_active = mat(n_active_, n_active_);
	for(u32 i = 0; i < n_active_; i++) {
	  for(u32 j = 0; j < n_active_; j++) {
	    Gram_active(i,j) = Gram_(active_set_[i], active_set_[j]);
	  }
	}
	
	unnormalized_beta_direction = 
	  solve(Gram_active % trans(S) % S, ones<mat>(n_active_, 1));
	normalization = 1.0 / sqrt(sum(unnormalized_beta_direction));
	beta_direction = normalization * unnormalized_beta_direction % s;
      }
      
      // compute "equiangular" direction in output space
      vec mu_direction = zeros(n_);
      for(u32 i = 0; i < n_active_; i++) {
	mu_direction += beta_direction(i) * X_.col(active_set_[i]);
      }
      

      double gamma = max_corr / normalization;
      change_ind = -1;
      // if not all variables are active
      if(n_active_ < p_) {
	// compute correlations with direction
	for(u32 ind = 0; ind < p_; ind++) {
	  if(is_active_[ind]) {
	    continue;
	  }
	  double dir_corr = dot(X_.col(ind), mu_direction);
	  double val1 = (max_corr - corr(ind)) / (normalization - dir_corr);
	  double val2 = (max_corr + corr(ind)) / (normalization + dir_corr);
	  if((val1 > 0) && (val1 < gamma)) {
	    gamma = val1;
	    change_ind = ind;
	  }
	  if((val2 > 0) && (val2 < gamma)) {
	    gamma = val2;
	    change_ind = ind;
	  }
	}
      }
      
      
      // bound gamma according to LASSO
      if(lasso_) {
	double lasso_bound_on_gamma = DBL_MAX;
	u32 active_ind_to_kick_out = -1;
	for(u32 i = 0; i < n_active_; i++) {
	  double val = -beta(active_set_[i]) / beta_direction(i);
	  if((val > 0) && (val < lasso_bound_on_gamma)) {
	    lasso_bound_on_gamma = val;
	    active_ind_to_kick_out = i;
	  }
	}

	if(lasso_bound_on_gamma < gamma) {
	  kick_out = true;
	  gamma = lasso_bound_on_gamma;
	  change_ind = active_ind_to_kick_out;
	}
      }
      
      // update prediction
      mu += gamma * mu_direction;
      
      // update estimator
      for(u32 i = 0; i < n_active_; i++) {
	beta(active_set_[i]) += gamma * beta_direction(i);
      }
      beta_path_.push_back(beta);
      
      // compute correlates
      corr = Xty_ - trans(X_) * mu;
      max_corr -= gamma * normalization;
      lambda_path_.push_back(max_corr);
      
      
      
      if(lasso_) {
	double ultimate_lambda = max_corr;
	if(ultimate_lambda <= desired_lambda_) {
	  InterpolateBeta(ultimate_lambda);
	  /*  
	  printf("ultimate_lambda = %f\ndesired_lambda = %f\n",
		 ultimate_lambda,
		 desired_lambda_);
	  int path_length = beta_path_.size();
	  
	  // interpolate beta and stop
	  double penultimate_lambda = lambda_path_[path_length - 2];
	  double interp = 
	    (penultimate_lambda - desired_lambda_)
	    / (penultimate_lambda - ultimate_lambda);
	  beta_path_[path_length - 1] = 
	    (1 - interp) * (beta_path_[path_length - 2]) 
	    + interp * beta_path_[path_length - 1];
	  lambda_path_[path_length - 1] = desired_lambda_;
	  */
	  break;
	}
      }
    }
    
  }

  
  
  void Deactive(u32 active_var_ind) {
    n_active_--;
    is_active_[active_set_[active_var_ind]] = 0;
    active_set_.erase(active_set_.begin() + active_var_ind);
  }
  
  void InterpolateBeta(double ultimate_lambda) {
    printf("ultimate_lambda = %f\ndesired_lambda = %f\n",
	   ultimate_lambda,
	   desired_lambda_);
    int path_length = beta_path_.size();
    
    // interpolate beta and stop
    double penultimate_lambda = lambda_path_[path_length - 2];
    double interp = 
      (penultimate_lambda - desired_lambda_)
      / (penultimate_lambda - ultimate_lambda);
    beta_path_[path_length - 1] = 
      (1 - interp) * (beta_path_[path_length - 2]) 
      + interp * beta_path_[path_length - 1];
    lambda_path_[path_length - 1] = desired_lambda_; 
  }
  
  
  void CholeskyInsert(mat& R, const vec& new_x, const mat& X) {
    if(R.n_rows == 0) {
      R = mat(1, 1);
      R(0,0) = norm(new_x, 2);
    }
    else {
      vec new_Gram_col = trans(X) * new_x;
      CholeskyInsert(R, new_x, new_Gram_col);
    }
  }
  
  
  void CholeskyInsert(mat& R, const vec& new_x, const vec& new_Gram_col) {
    int n = R.n_rows;
    
    if(n == 0) {
      R = mat(1, 1);
      R(0, 0) = norm(new_x, 2);
    }
    else {
      mat new_R = mat(n + 1, n + 1);
      
      vec R_k = solve(trimatl(trans(R)), new_Gram_col);
      double R_kk = sqrt(dot(new_x, new_x) - dot(R_k, R_k));
      
      new_R(span(0, n - 1), span(0, n - 1)) = R(span::all, span::all);
      new_R(span(0, n - 1), n) = R_k;
      new_R(n, span(0, n - 1)).fill(0.0);
      new_R(n, n) = R_kk;
      
      R = new_R;
    }
  }
  
  
  void GivensRotate(const vec& x, vec& rotated_x, mat& G) {
    if(x(1) == 0) {
      G = eye(2, 2);
      rotated_x = x;
    }
    else {
      double r = norm(x, 2);
      G = mat(2, 2);
      
      double scaled_x1 = x(0) / r;
      double scaled_x2 = x(1) / r;

      G(0,0) = scaled_x1;
      G(1,0) = -scaled_x2;
      G(0,1) = scaled_x2;
      G(1,1) = scaled_x1;
      
      rotated_x = vec(2);
      rotated_x(0) = r;
      rotated_x(1) = 0;
    }
  }
  
  
  void CholeskyDelete(mat& R, u32 col_to_kill) {
    printf("calling CholeskyDelete\n");
    u32 n = R.n_rows;
    
    if(col_to_kill == (n - 1)) {
      R = R(span(0, n - 2), span(0, n - 2));
    }
    else {
      R.shed_col(col_to_kill); // remove column col_to_kill
      n--;
      
      for(u32 k = col_to_kill; k < n; k++) {
	mat G;
	vec rotated_vec;
	GivensRotate(R(span(k, k + 1), k),
		     rotated_vec,
		     G);
	R(span(k, k + 1), k) = rotated_vec;
	if(k < n - 1) {
	  R(span(k, k + 1), span(k + 1, n - 1)) =
	    G * R(span(k, k + 1), span(k + 1, n - 1));
	}
      }
      R.shed_row(n);
    }
  }
  
};

#endif
