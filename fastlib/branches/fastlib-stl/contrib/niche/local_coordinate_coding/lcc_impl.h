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
  D_ = randu(n_dims_, n_atoms_);
  for(u32 j = 0; j < n_atoms_; j++) {
    D_.col(j) /= norm(D_.col(j), 2);
  }
}


void LocalCoordinateCoding::DoLCC(u32 n_iterations) {
  OptimizeCode();
  return;
  uvec adjacencies = find(V_);
  printf("Objective function value: %f\n",
	 Objective(adjacencies));
  printf("\n%d nonzero entries in code V (%f%%)\n\n",
	 adjacencies.n_elem,
	 ((double)(adjacencies.n_elem)) / ((double)(n_atoms_ * n_points_)));
  printf("Optimizing Codebook\n");
  OptimizeCodebook(adjacencies);
  printf("Objective function value: %f\n",
	 Objective(adjacencies));
}


void LocalCoordinateCoding::OptimizeCode() {
  mat sq_dists = 
    repmat(trans(sum(square(D_))), 1, n_points_)
    + repmat(sum(square(X_)), n_atoms_, 1)
    - 2 * trans(D_) * X_;
  
  mat inv_sq_dists = 1.0 / sq_dists; // is this allowed? it's so convenient!
  //return;
  
  for(u32 i = 0; i < n_points_; i++) {
    if((i % 1000) == 0) {
      printf(" %d", i);
    }
    vec w = sq_dists.col(i);
    vec inv_w = inv_sq_dists.col(i);
    mat D_prime = D_ * diagmat(inv_w);
    
    Lars lars;
    lars.Init(D_prime, X_.col(i), true, 0.5 * lambda_); // do we still need 0.5 * lambda??
    lars.DoLARS();
    vec beta;
    lars.Solution(beta);
    V_.col(i) = beta % inv_w;
  }
}


void LocalCoordinateCoding::OptimizeCodebook(uvec adjacencies) {
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
    if(((uvec)(find(V_.row(j)))).n_elem == 0) {
      inactive_atoms.push_back(j);
    }
    else {
      active_atoms.push_back(j);
    }
  }
  u32 n_active_atoms = active_atoms.size();
  u32 n_inactive_atoms = inactive_atoms.size();

  mat active_V;
  if(inactive_atoms.empty()) {
    active_V = V_;
  }
  else {
    active_V.set_size(n_active_atoms, n_points_);

    u32 cur_row = 0;
    u32 inactive_atom_ind = 0;
    // first, check 0 to first inactive atom
    if(inactive_atoms[0] > 0) {
      // note that this implies that n_atoms_ > 1
      u32 height = inactive_atoms[0];
      active_V(span(cur_row, cur_row + height - 1), span::all) =
	V_(span(0, inactive_atoms[0] - 1), span::all);
      cur_row += height;
    }
    // now, check i'th inactive atom to (i + 1)'th inactive atom, until i = penultimate atom
    while(inactive_atom_ind < n_inactive_atoms - 1) {
      u32 height = 
	inactive_atoms[inactive_atom_ind + 1]
	- inactive_atoms[inactive_atom_ind]
	- 1;
      if(height > 0) {
	active_V(span(cur_row, cur_row + height - 1), 
		 span::all) =
	  V_(span(inactive_atoms[inactive_atom_ind] + 1,
		  inactive_atoms[inactive_atom_ind + 1] - 1), 
	     span::all);
	cur_row += height;
      }
      inactive_atom_ind++;
    }
    // now that i is last inactive atom, check last inactive atom to last atom
    if(inactive_atoms[inactive_atom_ind] < n_atoms_ - 1) {
      active_V(span(cur_row, n_active_atoms - 1), 
	       span::all) = 
	V_(span(inactive_atoms[inactive_atom_ind] + 1, n_atoms_ - 1), 
	   span::all);
    }
  }
  
  std::vector<u32> atom_reverse_lookup(n_atoms_);
  for(u32 i = 0; i < n_active_atoms; i++) { // use an iterator, too lazy right now..
    atom_reverse_lookup[active_atoms[i]] = i;
  }
  
  printf("%d inactive atoms\n", n_inactive_atoms);
  
  mat V_prime = zeros(n_active_atoms, n_points_ + adjacencies.n_elem);
  for(u32 i = 0; i < n_active_atoms; i++) {
    V_prime(span::all, span(0, n_points_ - 1)) = active_V;
  }
  
  vec w_squared = ones(n_points_ + adjacencies.n_elem, 1);
  printf("building up V_prime\n");
  for(u32 l = 0; l < adjacencies.n_elem; l++) {
    u32 atom_ind = adjacencies(l) % n_atoms_;
    u32 point_ind = (u32) (adjacencies(l) / n_atoms_);
    V_prime(atom_reverse_lookup[atom_ind], n_points_ + l) = 1.0;
    w_squared(n_points_ + l) = V_(atom_ind, point_ind); 
  }
  w_squared.subvec(n_points_, w_squared.n_elem - 1) = 
    lambda_ * abs(w_squared.subvec(n_points_, w_squared.n_elem - 1));
  
  printf("about to solve\n");
  mat D_estimate;
  if(inactive_atoms.empty()) {
    printf("solving\n");
    D_estimate = 
      trans(solve(V_prime * diagmat(w_squared) * trans(V_prime),
		  V_prime * diagmat(w_squared) * trans(X_)));
  }
  else {
    D_estimate = zeros(n_dims_, n_atoms_);
    printf("solving\n");
    mat D_active_estimate = 
      trans(solve(V_prime * diagmat(w_squared) * trans(V_prime),
		  V_prime * diagmat(w_squared) * trans(X_)));
    for(u32 i = 0; i < n_active_atoms; i++) {
      D_estimate.col(active_atoms[i]) = D_active_estimate.col(i);
    }
    for(u32 i = 0; i < n_inactive_atoms; i++) {
      vec new_atom = randn(n_dims_, 1);
      D_estimate.col(inactive_atoms[i]) = 
	new_atom / norm(new_atom, 2);
    }
  }
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


void LocalCoordinateCoding::PrintDictionary() {
  D_.print("Dictionary");
}

  
void LocalCoordinateCoding::GetDictionary(mat& D) {
  D = D_;
}


void LocalCoordinateCoding::PrintCoding() {
  V_.print("Coding matrix");
}

