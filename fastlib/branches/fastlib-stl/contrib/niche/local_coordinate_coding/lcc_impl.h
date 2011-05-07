#ifndef INSIDE_LCC_H
#error "This is not a public header file!"
#endif



void LocalCoordinateCoding::Init(const mat& X, u32 n_atoms, double lambda) {
  X_ = X;

  n_dims_ = X.n_rows;
  n_points_ = X.n_cols;

  n_atoms_ = n_atoms;
  D_ = mat(n_dims_, n_atoms_);
  V_ = mat(n_atoms_, n_points_);
  
  lambda_ = lambda;
}


void LocalCoordinateCoding::SetDictionary(mat D) {
  D_ = D;
}


void LocalCoordinateCoding::InitDictionary() {  
  RandomInitDictionary();
}


void LocalCoordinateCoding::InitDictionary(const char* dictionary_filename) {  
  char* dictionary_fullpath = (char*) malloc(160 * sizeof(char));
  sprintf(dictionary_fullpath,
	  "../contrib/niche/local_coordinate_coding/%s",
	  dictionary_filename);
  D_.load(dictionary_fullpath);
}


void LocalCoordinateCoding::RandomInitDictionary() {
  D_ = randn(n_dims_, n_atoms_);
  for(u32 j = 0; j < n_atoms_; j++) {
    D_.col(j) /= norm(D_.col(j), 2);
  }
}


void LocalCoordinateCoding::DataDependentRandomInitDictionary() {
  for(u32 i = 0; i < n_atoms_; i++) {
    vec D_i = D_.unsafe_col(i);
    RandomAtom(D_i);
  }
}


void LocalCoordinateCoding::RandomAtom(vec& atom) {
  atom.zeros();
  const u32 n_seed_atoms = 3;
  for(u32 i = 0; i < n_seed_atoms; i++) {
    atom +=  X_.col(rand() % n_points_);
  }
  atom /= ((double) n_seed_atoms);
  atom /= norm(atom, 2);
}


void LocalCoordinateCoding::DoLCC(u32 n_iterations) {
  for(u32 t = 1; t <= n_iterations; t++) {
    printf("Iteration %d of %d\n", t, n_iterations);
    OptimizeCode();
    uvec adjacencies = find(V_);
    printf("Objective function value: %f\n",
	   Objective(adjacencies));
    printf("\n%d nonzero entries in code V (%f%%)\n\n",
	   adjacencies.n_elem,
	   ((double)(adjacencies.n_elem)) / ((double)(n_atoms_ * n_points_)));
    printf("Optimizing Dictionary\n");
    OptimizeDictionary(adjacencies);
    
    printf("Objective function value: %f\n",
	   Objective(adjacencies));
  }
}


void LocalCoordinateCoding::OptimizeCode() {
  mat sq_dists = 
    repmat(trans(sum(square(D_))), 1, n_points_)
    + repmat(sum(square(X_)), n_atoms_, 1)
    - 2 * trans(D_) * X_;
  
  mat inv_sq_dists = 1.0 / sq_dists;
  
  mat DtD = trans(D_) * D_;
  mat D_prime_transpose_D_prime(DtD.n_rows, DtD.n_cols);
  
  for(u32 i = 0; i < n_points_; i++) {
    // report progress
    if((i % 1000) == 0) {
      printf("%d\n", i);
    }
    
    vec w = sq_dists.unsafe_col(i);
    vec inv_w = inv_sq_dists.unsafe_col(i);
    mat D_prime = D_ * diagmat(inv_w);
    
    mat D_prime_transpose_D_prime = diagmat(inv_w) * DtD * diagmat(inv_w);
    
    Lars lars;
    // do we still need 0.5 * lambda? yes, yes we do
    //lars.Init(D_prime.memptr(), X_.colptr(i), n_dims_, n_atoms_, true, 0.5 * lambda_); // apparently not as fast as using the below duo
                                                                                       // this may change, depending on the dimensionality and sparsity

    // the duo
    lars.Init(D_prime.memptr(), X_.colptr(i), n_dims_, n_atoms_, false, 0.5 * lambda_);
    lars.SetGram(D_prime_transpose_D_prime.memptr(), n_atoms_);
    
    lars.DoLARS();
    vec beta;
    lars.Solution(beta);
    V_.col(i) = beta % inv_w;
  }
}


void LocalCoordinateCoding::OptimizeDictionary(uvec adjacencies) {
  // count number of atomic neighbors for each point x^i
  vec neighbor_counts = zeros(n_points_, 1);
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
  
  // build X_prime := [X x^1 ... x^1 ... x^n ... x^n]
  // where each x^i is repeated for the number of neighbors x^i has
  mat X_prime = zeros(n_dims_, n_points_ + adjacencies.n_elem);
  X_prime(span::all, span(0, n_points_ - 1)) = X_;
  u32 cur_col = n_points_;
  for(u32 i = 0; i < n_points_; i++) {
    if(neighbor_counts(i) > 0) {
      X_prime(span::all, span(cur_col, cur_col + neighbor_counts(i) - 1)) =
	repmat(X_.col(i), 1, neighbor_counts(i));
    }
    cur_col += neighbor_counts(i);
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
  
  mat V_prime = zeros(n_active_atoms, n_points_ + adjacencies.n_elem);
  printf("adjacencies.n_elem = %d\n", adjacencies.n_elem);
  V_prime(span::all, span(0, n_points_ - 1)) = active_V;
  
  vec w_squared = ones(n_points_ + adjacencies.n_elem, 1);
  printf("building up V_prime\n");
  for(u32 l = 0; l < adjacencies.n_elem; l++) {
    u32 atom_ind = adjacencies(l) % n_atoms_;
    u32 point_ind = (u32) (adjacencies(l) / n_atoms_);
    V_prime(atom_reverse_lookup(atom_ind), n_points_ + l) = 1.0;
    w_squared(n_points_ + l) = V_(atom_ind, point_ind); 
  }
  
  w_squared.subvec(n_points_, w_squared.n_elem - 1) = 
    lambda_ * abs(w_squared.subvec(n_points_, w_squared.n_elem - 1));
  
  printf("about to solve\n");
  mat D_estimate;
  if(inactive_atoms.empty()) {
    mat A = V_prime * diagmat(w_squared) * trans(V_prime);
    mat B = V_prime * diagmat(w_squared) * trans(X_prime);
    
    printf("solving\n");
    D_estimate = 
      trans(solve(A, B));
    /*    
    D_estimate = 
      trans(solve(V_prime * diagmat(w_squared) * trans(V_prime),
		  V_prime * diagmat(w_squared) * trans(X_prime)));
    */
  }
  else {
    D_estimate = zeros(n_dims_, n_atoms_);
    printf("solving\n");
    mat D_active_estimate = 
      trans(solve(V_prime * diagmat(w_squared) * trans(V_prime),
		  V_prime * diagmat(w_squared) * trans(X_prime)));
    for(u32 i = 0; i < n_active_atoms; i++) {
      D_estimate.col(active_atoms[i]) = D_active_estimate.col(i);
    }
    for(u32 i = 0; i < n_inactive_atoms; i++) {
      vec D_i = D_estimate.unsafe_col(inactive_atoms[i]);
      RandomAtom(D_i);
      /*
      vec new_atom = randn(n_dims_, 1);
      D_estimate.col(inactive_atoms[i]) = 
	new_atom / norm(new_atom, 2);
      */
    }
  }
  D_ = D_estimate;
}
// need to test above function, sleepy now, will resume soon!


double LocalCoordinateCoding::Objective(uvec adjacencies) {
  double first_part = 0;
  u32 n_adjacencies = adjacencies.n_elem;
  for(u32 l = 0; l < n_adjacencies; l++) {
    u32 atom_ind = adjacencies(l) % n_atoms_;
    u32 point_ind = (u32) (adjacencies(l) / n_atoms_);
    first_part += fabs(V_(atom_ind, point_ind)) * as_scalar(sum(square(D_.col(atom_ind) - X_.col(point_ind))));
  }
  double second_part = norm(X_ - D_ * V_, "fro");
  return lambda_ * first_part + second_part * second_part;
}


void LocalCoordinateCoding::GetDictionary(mat& D) {
  D = D_;
}


void LocalCoordinateCoding::PrintDictionary() {
  D_.print("Dictionary");
}


void LocalCoordinateCoding::GetCoding(mat& V) {
  V = V_;
}


void LocalCoordinateCoding::PrintCoding() {
  V_.print("Coding matrix");
}

