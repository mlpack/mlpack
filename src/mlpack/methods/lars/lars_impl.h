#ifndef INSIDE_LARS_H
#error "This is not a public header file!"
#endif

Lars::Lars() {
  lasso_ = false;
  elastic_net_ = false;
}

//~Lars() { }


// power user Init functions
void Lars::Init(double* X_mem, double* y_mem, u32 n, u32 p,
		bool use_cholesky, double lambda_1, double lambda_2) {
  X_ = mat(X_mem, n, p, false, true);
  y_ = vec(y_mem, n, false, true);
  
  n_ = n;
  p_ = p;
  
  lasso_ = true;
  lambda_1_ = lambda_1;
  elastic_net_ = true;
  lambda_2_ = lambda_2;
  use_cholesky_ = use_cholesky;
}

void Lars::Init(double* X_mem, double* y_mem, u32 n, u32 p,
		bool use_cholesky, double lambda_1) {
  X_ = mat(X_mem, n, p, false, true);
  y_ = vec(y_mem, n, false, true);
  
  n_ = n;
  p_ = p;
  
  lasso_ = true;
  lambda_1_ = lambda_1;
  use_cholesky_ = use_cholesky;
}

void Lars::Init(double* X_mem, double* y_mem, u32 n, u32 p,
		bool use_cholesky) {
  X_ = mat(X_mem, n, p, false, true);
  y_ = vec(y_mem, n, false, true);
  
  n_ = n;
  p_ = p;
  
  use_cholesky_ = use_cholesky;
}


// normal Init functions
void Lars::Init(const mat& X, const vec& y,
		bool use_cholesky, double lambda_1, double lambda_2) {
  X_ = mat(X);
  y_ = vec(y);
  
  n_ = X_.n_rows;
  p_ = X_.n_cols;
  
  
  lasso_ = true;
  lambda_1_ = lambda_1;
  elastic_net_ = true;
  lambda_2_ = lambda_2;
  use_cholesky_ = use_cholesky;
}

void Lars::Init(const mat& X, const vec& y,
		bool use_cholesky, double lambda_1) {
  X_ = mat(X);
  y_ = vec(y);
  
  n_ = X_.n_rows;
  p_ = X_.n_cols;

  lasso_ = true;
  lambda_1_ = lambda_1;
  use_cholesky_ = use_cholesky;
}

void Lars::Init(const mat& X, const vec& y, bool use_cholesky) {
  X_ = mat(X);
  y_ = vec(y);
  
  n_ = X_.n_rows;
  p_ = X_.n_cols;
  
  use_cholesky_ = use_cholesky;
}

/*
void Lars::CommonInit() {
  ComputeXty();
  if(!use_cholesky_ && Gram_.is_empty()) {
    ComputeGram();
  }
  
  // set up active set variables
  n_active_ = 0;
  active_set_ = std::vector<u32>(0);
  is_active_ = std::vector<bool>(p_);
  fill(is_active_.begin(), is_active_.end(), false);
}
*/

void Lars::SetGram(double* Gram_mem, u32 p) {
  Gram_ = mat(Gram_mem, p, p, false, true);
}

void Lars::SetGram(const mat& Gram) {
  Gram_ = Gram;
}


void Lars::ComputeGram() {
  if(elastic_net_) {
    Gram_ = trans(X_) * X_ + lambda_2_ * eye(p_, p_);
  }
  else {
    Gram_ = trans(X_) * X_;
  }
}


void Lars::ComputeXty() {
  Xty_ = trans(X_) * y_;
}    


void Lars::UpdateX(const std::vector<int>& col_inds, const mat& new_cols) {
  for(u32 i = 0; i < col_inds.size(); i++) {
    X_.col(col_inds[i]) = new_cols.col(i);
  }
  
  if(!use_cholesky_) {
    UpdateGram(col_inds);
  }
  UpdateXty(col_inds);
}

void Lars::UpdateGram(const std::vector<int>& col_inds) {
  for (std::vector<int>::const_iterator i = col_inds.begin(); 
       i != col_inds.end(); 
       ++i) {
    for (std::vector<int>::const_iterator j = col_inds.begin(); 
	 j != col_inds.end(); 
	 ++j) {
      Gram_(*i, *j) = dot(X_.col(*i), X_.col(*j));
    }
  }
    
  if(elastic_net_) {
    for (std::vector<int>::const_iterator i = col_inds.begin(); 
	 i != col_inds.end(); 
	 ++i) {
      Gram_(*i, *i) += lambda_2_;
    }
  }
}
  
  
void Lars::UpdateXty(const std::vector<int>& col_inds) {
  for (std::vector<int>::const_iterator i = col_inds.begin(); 
       i != col_inds.end(); 
       ++i) {
    Xty_(*i) = dot(X_.col(*i), y_);
  }
}
  
  
  
void Lars::PrintGram() {
  Gram_.print("Gram matrix");
}
  
  
void Lars::SetY(const vec& y) {
  y_ = y;
}
  
  
void Lars::PrintY() {
  y_.print();
}


const std::vector<u32> Lars::active_set() {
  return active_set_;
}


const std::vector<vec> Lars::beta_path() {
  return beta_path_;
}

  
const std::vector<double> Lars::lambda_path() {
  return lambda_path_;
}
  
  
void Lars::SetDesiredLambda(double lambda_1) {
  lambda_1_ = lambda_1;
}
  
  
void Lars::DoLARS() {
  // compute Gram matrix, XtY, and initialize active set varibles
  ComputeXty();
  if(!use_cholesky_ && Gram_.is_empty()) {
    ComputeGram();
  }
  
  // set up active set variables
  n_active_ = 0;
  active_set_ = std::vector<u32>(0);
  is_active_ = std::vector<bool>(p_);
  fill(is_active_.begin(), is_active_.end(), false);
  
  
  
  // initialize y_hat and beta
  vec beta = zeros(p_);
  vec y_hat = zeros(n_);
  vec y_hat_direction = vec(n_);
    
  bool lasso_cond = false;

  // used for elastic net
  double sqrt_lambda_2 = -1;
  if(elastic_net_) {
    sqrt_lambda_2 = sqrt(lambda_2_);
  }
  else {
    lambda_2_ = -1; // for no particular reason
  }
    
  vec corr = Xty_;
  vec abs_corr = abs(corr);
  u32 change_ind;
  double max_corr = abs_corr.max(change_ind); // change_ind gets set here
    
  beta_path_.push_back(beta);
  lambda_path_.push_back(max_corr);

  // don't even start!
  if(max_corr < lambda_1_) {
    lambda_path_[0] = lambda_1_;
    return;
  }
    
  //u32 n_iterations_run = 0;
  // MAIN LOOP
  while((n_active_ < p_) && (max_corr > EPS)) {
    
    //n_iterations_run++;
    //printf("iteration %d\t", n_iterations_run);

    // explicit computation of max correlation, among inactive indices
    change_ind = -1;
    max_corr = 0;
    for(u32 i = 0; i < p_; i++) {
      if(!is_active_[i]) {
	if(fabs(corr(i)) > max_corr) {
	  max_corr = fabs(corr(i));
	  change_ind = i;
	}
      }
    }
    
    if(!lasso_cond) {
      // index is absolute index
      //printf("activating %d\n", change_ind);
      
      if(use_cholesky_) {
	vec new_Gram_col = vec(n_active_);
	for(u32 i = 0; i < n_active_; i++) {
	  new_Gram_col[i] = 
	    dot(X_.col(active_set_[i]), X_.col(change_ind));
	}
	CholeskyInsert(X_.col(change_ind), new_Gram_col);
      }
	
      // add variable to active set
      Activate(change_ind);
    }
    

    // compute signs of correlations
    vec s = vec(n_active_);
    for(u32 i = 0; i < n_active_; i++) {
      s(i) = corr(active_set_[i]) / fabs(corr(active_set_[i]));
    }

    
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
      unnormalized_beta_direction = solve(trimatu(R_), 
					  solve(trimatl(trans(R_)), s));

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
      
      mat S = s * ones<mat>(1, n_active_);
      unnormalized_beta_direction = 
	solve(Gram_active % trans(S) % S, ones<mat>(n_active_, 1));
      normalization = 1.0 / sqrt(sum(unnormalized_beta_direction));
      beta_direction = normalization * unnormalized_beta_direction % s;
    }
    
    // compute "equiangular" direction in output space
    ComputeYHatDirection(beta_direction, y_hat_direction);
    
    
    double gamma = max_corr / normalization;
    
    // if not all variables are active
    if(n_active_ < p_) {
      // compute correlations with direction
      for(u32 ind = 0; ind < p_; ind++) {
	if(is_active_[ind]) {
	  continue;
	}
	double dir_corr = dot(X_.col(ind), y_hat_direction);
	double val1 = (max_corr - corr(ind)) / (normalization - dir_corr);
	double val2 = (max_corr + corr(ind)) / (normalization + dir_corr);
	if((val1 > 0) && (val1 < gamma)) {
	  gamma = val1;
	}
	if((val2 > 0) && (val2 < gamma)) {
	  gamma = val2;
	}
      }
    }
    
    // bound gamma according to LASSO
    if(lasso_) {
      lasso_cond = false;
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
	//printf("%d: gap = %e\tbeta(%d) = %e\n", active_set_[active_ind_to_kick_out], gamma - lasso_bound_on_gamma, active_set_[active_ind_to_kick_out], beta(active_set_[active_ind_to_kick_out]));
	gamma = lasso_bound_on_gamma;
	lasso_cond = true;
	change_ind = active_ind_to_kick_out;
      }
    }
      
    // update prediction
    y_hat += gamma * y_hat_direction;
      
    // update estimator
    for(u32 i = 0; i < n_active_; i++) {
      beta(active_set_[i]) += gamma * beta_direction(i);
    }
    beta_path_.push_back(beta);

    
    if(lasso_cond) {
      // index is in position change_ind in active_set
      //printf("\t\tKICK OUT %d!\n", active_set_[change_ind]);
      if(beta(active_set_[change_ind]) != 0) {
	//printf("fixed from %e to 0\n", beta(active_set_[change_ind]));
	beta(active_set_[change_ind]) = 0;
      }
      if(use_cholesky_) {
	CholeskyDelete(change_ind);
      }
      Deactivate(change_ind);
    }
    
    corr = Xty_ - trans(X_) * y_hat;
    if(elastic_net_) {
      corr -= lambda_2_ * beta;
    }
    double cur_lambda = 0;
    for(u32 i = 0; i < n_active_; i++) {
      cur_lambda += fabs(corr(active_set_[i]));
    }
    cur_lambda /= ((double)n_active_);
    
    lambda_path_.push_back(cur_lambda);
    
    // Time to stop for LASSO?
    if(lasso_) {
      if(cur_lambda <= lambda_1_) {
	InterpolateBeta();
	break;
      }
    }
  }
    
}

  
void Lars::Solution(vec& beta) {
  beta = beta_path().back();
}


void Lars::GetCholFactor(mat& R) {
  R = R_;
}
  
  
void Lars::Deactivate(u32 active_var_ind) {
  n_active_--;
  is_active_[active_set_[active_var_ind]] = false;
  active_set_.erase(active_set_.begin() + active_var_ind);
}
  

void Lars::Activate(u32 var_ind) {
  n_active_++;
  is_active_[var_ind] = true;
  active_set_.push_back(var_ind);
}

  
void Lars::ComputeYHatDirection(const vec& beta_direction,
				vec& y_hat_direction) {
  y_hat_direction.fill(0);
  for(u32 i = 0; i < n_active_; i++) {
    y_hat_direction += beta_direction(i) * X_.col(active_set_[i]);
  }
}
  
  
void Lars::InterpolateBeta() {
  int path_length = beta_path_.size();
    
  // interpolate beta and stop
  double ultimate_lambda = lambda_path_[path_length - 1];
  double penultimate_lambda = lambda_path_[path_length - 2];
  double interp = 
    (penultimate_lambda - lambda_1_)
    / (penultimate_lambda - ultimate_lambda);
  beta_path_[path_length - 1] = 
    (1 - interp) * (beta_path_[path_length - 2]) 
    + interp * beta_path_[path_length - 1];
  lambda_path_[path_length - 1] = lambda_1_; 
}
  
  
void Lars::CholeskyInsert(const vec& new_x, const mat& X) {
  if(R_.n_rows == 0) {
    R_ = mat(1, 1);
    if(elastic_net_) {
      R_(0, 0) = sqrt(dot(new_x, new_x) + lambda_2_);
    }
    else {
      R_(0, 0) = norm(new_x, 2);
    }
  }
  else {
    vec new_Gram_col = trans(X) * new_x;
    CholeskyInsert(new_x, new_Gram_col);
  }
}
  
  
void Lars::CholeskyInsert(const vec& new_x, const vec& new_Gram_col) {
  int n = R_.n_rows;
    
  if(n == 0) {
    R_ = mat(1, 1);
    if(elastic_net_) {
      R_(0, 0) = sqrt(dot(new_x, new_x) + lambda_2_);
    }
    else {
      R_(0, 0) = norm(new_x, 2);
    }
  }
  else {
    mat new_R = mat(n + 1, n + 1);
      
    double sq_norm_new_x;
    if(elastic_net_) {
      sq_norm_new_x = dot(new_x, new_x) + lambda_2_;
    }
    else {
      sq_norm_new_x = dot(new_x, new_x);
    }
      
    vec R_k = solve(trimatl(trans(R_)), new_Gram_col);
    
    new_R(span(0, n - 1), span(0, n - 1)) = R_;//(span::all, span::all);
    new_R(span(0, n - 1), n) = R_k;
    new_R(n, span(0, n - 1)).fill(0.0);
    new_R(n, n) = sqrt(sq_norm_new_x - dot(R_k, R_k));
      
    R_ = new_R;
  }
}
  
  
void Lars::GivensRotate(const vec& x, vec& rotated_x, mat& G) {
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
  
  
void Lars::CholeskyDelete(u32 col_to_kill) {
  u32 n = R_.n_rows;
    
  if(col_to_kill == (n - 1)) {
    R_ = R_(span(0, n - 2), span(0, n - 2));
  }
  else {
    R_.shed_col(col_to_kill); // remove column col_to_kill
    n--;
      
    for(u32 k = col_to_kill; k < n; k++) {
      mat G;
      vec rotated_vec;
      GivensRotate(R_(span(k, k + 1), k),
		   rotated_vec,
		   G);
      R_(span(k, k + 1), k) = rotated_vec;
      if(k < n - 1) {
	R_(span(k, k + 1), span(k + 1, n - 1)) =
	  G * R_(span(k, k + 1), span(k + 1, n - 1));
      }
    }
    R_.shed_row(n);
  }
}
