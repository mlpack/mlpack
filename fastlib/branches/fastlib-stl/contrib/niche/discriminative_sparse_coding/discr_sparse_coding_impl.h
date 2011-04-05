#ifndef INSIDE_DISCR_SPARSE_CODING_H
#error "This is not a public header file!"
#endif

void DiscrSparseCoding::Init(const mat& X, const vec& y, u32 n_atoms,
			     double lambda_1, double lambda_2,
			     double lambda_w) {
  X_ = X;
  y_ = y;
  
  n_dims_ = X.n_rows;
  n_points_ = X.n_cols;
  
  n_atoms_ = n_atoms;
  D_ = mat(n_dims_, n_atoms_);
  
  w_ = vec(n_atoms_);
  
  lambda_1_ = lambda_1;
  lambda_2_ = lambda_2;
  lambda_w_ = lambda_w;
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
    SGDStep(X_.col(ind), y_(ind), step_size);
    // modify step size in some way
  }
}


void DiscrSparseCoding::SGDStep(const vec& x, double y, double step_size) {
  Lars lars;
  lars.Init(D_, x, true, lambda_1_, lambda_2_);
  lars.DoLARS();
  vec v;
  lars.Solution(v);
  
  if(y * dot(v, w_) > 1) {
    // no update necessary
    return;
  }

  mat chol_factor;
  lars.GetCholFactor(chol_factor);
  
  // active set
  std::vector<u32> active_set = lars.active_set();
  u32 n_active = active_set.size();
  
  // for the update, we need (D_Lambda^T + D_Lambda)^{-1}
  // fortunately, we already have the cholesky factorization of this in lars
  
  // first, set up some things that will be useful later
  vec w_active = vec(n_active);
  vec D_active = mat(n_dims_, n_active);
  vec v_active = vec(n_active);
  vec sign_v_active = vec(n_active);

  for(u32 i = 0; i < n_active; i++) {
    u32 ind = active_set[i];
    w_active(i) = w_(ind);
    D_active.col(i) = D_.col(ind);
    double cur_v = v(ind);
    v_active(i) = cur_v;
    sign_v_active(i) = cur_v / fabs(cur_v);
  }
  
  
  // first, update D_active
  
  // Let A := inv(D^T * D + lambda_2 * I)
  
  // 4 parts
  // 1st part:
  vec A_w = solve(chol_factor, solve(trans(chol_factor), w_active));
  vec wt_A = trans(A_w);
  
  vec first_part_lhs = D_active;
  vec first_part_middle = A_w;
  vec first_part_rhs = trans(v_active);
  mat first_part = first_part_lhs * first_part_middle * first_part_rhs;
  //
  // 2nd part:
  vec second_part_lhs = D_active * v_active;
  vec second_part_rhs = wt_A;
  mat second_part = second_part_lhs * second_part_rhs;
  //
  // 3rd part:
  vec third_part_lhs = x;
  vec third_part_rhs = wt_A;
  mat third_part = -third_part_lhs * third_part_rhs;
  //
  // 4th part:
  vec fourth_part_1_lhs = x;
  vec fourth_part_1_rhs = wt_A % sign_v_active;
  mat fourth_part_2 = D_active;
  vec fourth_part_2_col_scaling = wt_A % sign_v_active;
  mat fourth_part = 
    -fourth_part_1_lhs * fourth_part_1_rhs - 
    fourth_part_2 * diagmat(fourth_part_2_col_scaling);

  mat D_active_update = step_size * y * (first_part + second_part + third_part + fourth_part);

  
  // now, update hypothesis vector w
  w_ += step_size * -y * v;
  
  
  // now, update dictionary D using D_active
  // also, project each modified column onto unit ball
  for(u32 i = 0; i < n_active; i++) {
    u32 ind = active_set[i];
    D_.col(active_set[i]) += D_active_update.col(i);
    double norm_D_ind = norm(D_.col(ind), 2);
    if(norm_D_ind > 1) {
      D_.col(ind) /= norm_D_ind;
    }
  }
  
  ProjectW();
}


void DiscrSparseCoding::ProjectW() {
  double norm_w = norm(w_, 2);
  // can we use 1 / lambda_w, as in Pegasos, or can we only use the weaker 2 / lambda_w?
  if(norm_w > 2 / lambda_w_) {
    w_ = w_  * ((2 / lambda_w_) / norm_w);
  }
}
