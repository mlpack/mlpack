#ifndef INSIDE_DISCR_SPARSE_CODING_H
#error "This is not a public header file!"
#endif

void DiscrSparseCoding::Init(const mat& X, const vec& y, u32 n_atoms,
			     double lambda_1, double lambda_2) {
  X_ = X;
  y_ = y;
  
  n_dims_ = X.n_rows;
  n_points_ = X.n_cols;
  
  n_atoms_ = n_atoms;
  D_ = mat(n_dims_, n_atoms_);
  
  w_ = vec(n_atoms_);
  
  lambda_1_ = lambda_1;
  lambda_2_ = lambda_2;
}


void DiscrSparseCoding::InitDictionary() {
  RandomInitDictionary();
}


void DiscrSparseCoding::RandomInitDictionary() {
  for(u32 j = 0; j < n_atoms_; j++) {
    D_.col(j) = randu(n_dims_);
    D_.col(j) /= norm(D_.col(j), 2);
  }
}


void DiscrSparseCoding::KMeansInitDictionary() {
  // need a constrained k-means algorithm to ensure each cluster is assigned at least one point
}


void DiscrSparseCoding::InitW() {
  w_.zeros();
}


void DiscrSparseCoding::SGDOptimize(u32 n_iterations, double step_size) {
  for(u32 t = 0; t < n_iterations; t++) {
    u32 ind = rand() % n_points_;
    SGDStep(X_.col(ind), step_size);
    // modify step size in some way
  }
}


void DiscrSparseCoding::SGDStep(const vec& x, double step_size) {
  Lars lars;
  lars.Init(D_, x, true, lambda_1_, lambda_2_);
  lars.DoLARS();
  vec v;
  lars.Solution(v);
  mat A_chol_factor;
  lars.GetCholFactor(A_chol_factor);
  
  // active set
  std::vector<u32> active_set = lars.active_set();
  u32 n_active = active_set.size();
  
  // for the update, we need (D_Lambda^T + D_Lambda)^{-1}
  // fortunately, we already have the cholesky factorization of this in lars
    
  for(u32 i = 0; i < n_active; i++) {
    
    
  }
  
  // update to Dictionary
  
}
