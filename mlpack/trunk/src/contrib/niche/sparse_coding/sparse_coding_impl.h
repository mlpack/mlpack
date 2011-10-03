#ifndef INSIDE_SPARSE_CODING_H
#error "This is not a public header file!"
#endif

#define NORM_GRAD_TOL 1e-6

void SparseCoding::Init(double* X_mem, u32 n_dims, u32 n_points,
			u32 n_atoms, double lambda) {
  X_ = mat(X_mem, n_dims, n_points, false, true);

  n_dims_ = n_dims;
  n_points_ = n_points;

  n_atoms_ = n_atoms;
  //D_ = mat(n_dims_, n_atoms_);
  V_ = mat(n_atoms_, n_points_);
  
  lambda_ = lambda;
}


void SparseCoding::Init(const mat& X, u32 n_atoms, double lambda) {
  X_ = X;

  n_dims_ = X.n_rows;
  n_points_ = X.n_cols;

  n_atoms_ = n_atoms;
  //D_ = mat(n_dims_, n_atoms_);
  V_ = mat(n_atoms_, n_points_);
  
  lambda_ = lambda;
}


void SparseCoding::SetDictionary(double* D_mem) {
  D_ = mat(D_mem, n_dims_, n_atoms_, false, true);
}


void SparseCoding::SetDictionary(const mat& D) {
  D_ = D;
}


void SparseCoding::InitDictionary() {  
  RandomInitDictionary();
}


void SparseCoding::InitDictionary(const char* dictionary_filename) {  
  D_.load(dictionary_filename);
}


void SparseCoding::RandomInitDictionary() {
  D_ = randn(n_dims_, n_atoms_);
  for(u32 j = 0; j < n_atoms_; j++) {
    D_.col(j) /= norm(D_.col(j), 2);
  }
}


void SparseCoding::DataDependentRandomInitDictionary() {
  D_ = mat(n_dims_, n_atoms_);
  for(u32 i = 0; i < n_atoms_; i++) {
    vec D_i = D_.unsafe_col(i);
    RandomAtom(D_i);
  }
}


void SparseCoding::RandomAtom(vec& atom) {
  atom.zeros();
  const u32 n_seed_atoms = 3;
  for(u32 i = 0; i < n_seed_atoms; i++) {
    atom +=  X_.col(rand() % n_points_);
  }
  atom /= ((double) n_seed_atoms);
  atom /= norm(atom, 2);
}


void SparseCoding::DoSparseCoding(u32 n_iterations) {
  for(u32 t = 1; t <= n_iterations; t++) {
    printf("Iteration %d of %d\n", t, n_iterations);
    printf("Optimizing Coding\n");
    OptimizeCode();
    uvec adjacencies = find(V_);
    printf("Objective function value: %f\n",
	   Objective(adjacencies));
    printf("\n%d nonzero entries in code V (%f%%)\n\n",
	   adjacencies.n_elem,
	   100.0 * ((double)(adjacencies.n_elem)) / ((double)(n_atoms_ * n_points_)));
    printf("Optimizing Dictionary\n");
    OptimizeDictionary(adjacencies);
    ProjectDictionary();
    printf("Objective function value: %f\n",
	   Objective(adjacencies));
  }
  printf("Final Optimization of Coding\n");
  OptimizeCode();
}


void SparseCoding::OptimizeCode() {
  mat DtD = trans(D_) * D_;
  
  for(u32 i = 0; i < n_points_; i++) {
    // report progress
    if((i % 1000) == 0) {
      printf("%d\n", i);
    }
    
    Lars lars;
    // do we still need 0.5 * lambda? yes, yes we do
    //lars.Init(D.memptr(), X_.colptr(i), n_dims_, n_atoms_, true, 0.5 * lambda_); // apparently not as fast as using the below duo
                                                                                       // this may change, depending on the dimensionality and sparsity

    // the duo
    lars.Init(D_.memptr(), X_.colptr(i), n_dims_, n_atoms_, false, 0.5 * lambda_);
    lars.SetGram(DtD.memptr(), n_atoms_);
    
    lars.DoLARS();
    vec beta;
    lars.Solution(beta);
    V_.col(i) = beta;
  }
}


void SparseCoding::OptimizeDictionary(uvec adjacencies) {
  // count number of atomic neighbors for each point x^i
  uvec neighbor_counts = zeros<uvec>(n_points_, 1);
  if(adjacencies.n_elem > 0) {
    // this gets the column index
    u32 cur_point_ind = (u32)(adjacencies(0) / n_atoms_);
    u32 cur_count = 1;
    for(u32 l = 1; l < adjacencies.n_elem; l++) {
      if((u32)(adjacencies(l) / n_atoms_) == cur_point_ind) {
	cur_count++;
      }
      else {
	neighbor_counts(cur_point_ind) = cur_count;
	cur_point_ind = (u32)(adjacencies(l) / n_atoms_);
	cur_count = 1;
      }
    }
    neighbor_counts(cur_point_ind) = cur_count;
  }
  
  // handle the case of inactive atoms (atoms not used in the given coding)
  std::vector<u32> inactive_atoms;
  std::vector<u32> active_atoms;
  active_atoms.reserve(n_atoms_);
  for(u32 j = 0; j < n_atoms_; j++) {
    if(accu(V_.row(j) != 0) == 0) {
      inactive_atoms.push_back(j);
    }
    else {
      active_atoms.push_back(j);
    }
  }
  u32 n_active_atoms = active_atoms.size();
  u32 n_inactive_atoms = inactive_atoms.size();

  // efficient construction of V restricted to active atoms
  mat active_V;
  if(inactive_atoms.empty()) {
    active_V = V_;
  }
  else {
    uvec inactive_atoms_vec = conv_to< uvec >::from(inactive_atoms);
    RemoveRows(V_, inactive_atoms_vec, active_V);
  }
  
  uvec atom_reverse_lookup = uvec(n_atoms_);
  for(u32 i = 0; i < n_active_atoms; i++) {
    atom_reverse_lookup(active_atoms[i]) = i;
  }

  
  printf("%d inactive atoms\n", n_inactive_atoms);
  
  printf("Solving Dual via Newton's Method\n");
  mat D_estimate;
  // solve using Newton's method in the dual - note that the final dot multiplication with inv(A) seems to be unavoidable. Although more expensive, the code written this way (we use solve()) should be more numerically stable than just using inv(A) for everything.
  vec dual_vars = zeros<vec>(n_active_atoms);
  //vec dual_vars = 1e-14 * ones<vec>(n_active_atoms);
  //vec dual_vars = 10.0 * randu(n_active_atoms, 1); // method used by feature sign code - fails miserably here. perhaps the MATLAB optimizer fmincon does something clever?
  /*vec dual_vars = diagvec(solve(D_, X_ * trans(V_)) - V_ * trans(V_));
  for(u32 i = 0; i < dual_vars.n_elem; i++) {
    if(dual_vars(i) < 0) {
      dual_vars(i) = 0;
    }
  }
  */
  //dual_vars.print("dual vars");

  bool converged = false;
  mat V_XT = active_V * trans(X_);
  mat V_VT = active_V * trans(active_V);
  for(u32 t = 1; !converged; t++) {
    mat A = V_VT + diagmat(dual_vars);
    //mat R = chol(A);
    
    //mat A_inv_V_XT = solve(trimatu(R), solve(trimatl(trans(R)), V_ * trans(X_)));
    //printf("solving for A_inv_V_XT...");
    
    mat A_inv_V_XT = solve(A, V_XT);
    //printf("\n");
    
    vec gradient = -( sum(square(A_inv_V_XT), 1) - ones<vec>(n_active_atoms) );
    
    mat hessian = 
      -( -2 * (A_inv_V_XT * trans(A_inv_V_XT)) % inv(A) );
    
    //printf("solving for dual variable update...");
    vec search_direction = -solve(hessian, gradient);
    //vec search_direction = -gradient;

 
    
    // BEGIN ARMIJO LINE SEARCH
    double c = 1e-4;
    double alpha = 1.0;
    double rho = 0.9;
    double sufficient_decrease = c * dot(gradient, search_direction);

    /*
    {
      double sum_dual_vars = sum(dual_vars);    
      double f_old = 
	-( -trace(trans(V_XT) * A_inv_V_XT) - sum_dual_vars );
      printf("f_old = %f\t", f_old);
      double f_new = 
	-( -trace(trans(V_XT) * solve(V_VT + diagmat(dual_vars + alpha * search_direction), V_XT))
	  - (sum_dual_vars + alpha * sum(search_direction)) );
      printf("f_new = %f\n", f_new);
    }
    */
    
    double improvement;
    while(true) {
      // objective
      double sum_dual_vars = sum(dual_vars);
      double f_old = 
	-( -trace(trans(V_XT) * A_inv_V_XT) - sum_dual_vars );
      double f_new = 
	-( -trace(trans(V_XT) * solve(V_VT + diagmat(dual_vars + alpha * search_direction), V_XT))
	   - (sum_dual_vars + alpha * sum(search_direction)) );
      /*
      printf("alpha = %e\n", alpha);
      printf("norm of gradient = %e\n", norm(gradient, 2));
      printf("sufficient_decrease = %e\n", sufficient_decrease);
      printf("f_new - f_old - sufficient_decrease = %e\n", 
	     f_new - f_old - alpha * sufficient_decrease);
      */
      if(f_new <= f_old + alpha * sufficient_decrease) {
	search_direction = alpha * search_direction;
	improvement = f_old - f_new;
	break;
      }
      alpha *= rho;
    }
    // END ARMIJO LINE SEARCH
    
    dual_vars += search_direction;
    //printf("\n");
    double norm_gradient = norm(gradient, 2);
    printf("Iteration %d: ", t);
    printf("norm of gradient = %e\n", norm_gradient);
    /*
      if(norm_gradient < NORM_GRAD_TOL) {
      converged = true;
      }*/
    printf("improvement = %e\n", improvement);
    if(improvement < NORM_GRAD_TOL) {
      converged = true;
    }
  }
  //dual_vars.print("dual solution");
  if(inactive_atoms.empty()) {
    D_estimate = trans(
		       solve(V_VT + diagmat(dual_vars),
			     V_XT)
		       );
  }
  else {
    mat D_active_estimate = trans(
				  solve(V_VT + diagmat(dual_vars),
					V_XT)
				  );
    D_estimate = zeros(n_dims_, n_atoms_);
    for(u32 i = 0; i < n_active_atoms; i++) {
      D_estimate.col(active_atoms[i]) = D_active_estimate.col(i);
    }
    for(u32 i = 0; i < n_inactive_atoms; i++) {
      vec D_i = D_estimate.unsafe_col(inactive_atoms[i]);
      RandomAtom(D_i);
    }
  }
  D_ = D_estimate;
}


void SparseCoding::ProjectDictionary() {
  for(u32 j = 0; j < n_atoms_; j++) {
    double norm_D_j = norm(D_.col(j), 2);
    if(norm_D_j > 1) {
      /*
      if(norm_D_j - 1.0 > 1e-10) {
	//printf("norm exceeded 1 by %e, shrinking...\n", norm_D_j - 1.0);
	D_.col(j) /= norm(D_.col(j), 2);
      }
      */
      D_.col(j) /= norm(D_.col(j), 2);
    }
  }
}


double SparseCoding::Objective(uvec adjacencies) {
  double first_part = 0;
  u32 n_adjacencies = adjacencies.n_elem;
  for(u32 l = 0; l < n_adjacencies; l++) {
    u32 atom_ind = adjacencies(l) % n_atoms_;
    u32 point_ind = (u32) (adjacencies(l) / n_atoms_);
    first_part += fabs(V_(atom_ind, point_ind));
  }
  double second_part = norm(X_ - D_ * V_, "fro");
  return lambda_ * first_part + second_part * second_part;
}


void SparseCoding::GetDictionary(mat& D) {
  D = D_;
}


void SparseCoding::PrintDictionary() {
  D_.print("Dictionary");
}


void SparseCoding::GetCoding(mat& V) {
  V = V_;
}


void SparseCoding::PrintCoding() {
  V_.print("Coding matrix");
}
