#include "fock_matrix_comparison.h"

void FockMatrixComparison::ComputeCoreMatrices_() {
  
  printf("=== Computing Core Matrices ===\n");
  
  Matrix kinetic_energy_integrals;
  kinetic_energy_integrals.Init(num_entries_, num_entries_);
  kinetic_energy_integrals.SetZero();
  Matrix potential_energy_integrals;
  potential_energy_integrals.Init(num_entries_, num_entries_);
  potential_energy_integrals.SetZero();
  
  
  ArrayList<BasisShell> shells;
  eri::CreateShells(centers_, exponents_, momenta_, &shells);
  
  // iterate over list, compute integrals, and sum them into matrix
  
  for (index_t i = 0; i < shells.size(); i++) {
    
    for (index_t j = 0; j < i; j++) {
      
      // compute overlap
      Vector overlap; 
      eri::ComputeOverlapIntegrals(shells[i], shells[j], &overlap);
      Matrix overlap_mat;
      overlap_mat.Copy(overlap.ptr(), shells[i].num_functions(), 
                       shells[j].num_functions());
      
      
      eri::AddSubmatrix(shells[i].matrix_indices(), shells[j].matrix_indices(),
                        overlap_mat, &overlap_matrix_);
      
      // now transpose and add below the diagonal
      Matrix overlap_trans;
      la::TransposeInit(overlap_mat, &overlap_trans);
      eri::AddSubmatrix(shells[j].matrix_indices(), shells[i].matrix_indices(), 
                        overlap_trans, &overlap_matrix_);
      
      // compute kinetic
      Vector kinetic;
      eri::ComputeKineticIntegrals(shells[i], shells[j], &kinetic);
      
      Matrix kinetic_mat;
      kinetic_mat.Copy(kinetic.ptr(), shells[i].num_functions(), 
                       shells[j].num_functions());
      
      eri::AddSubmatrix(shells[i].matrix_indices(), shells[j].matrix_indices(),
                        kinetic_mat, &kinetic_energy_integrals);
      
      Matrix kinetic_trans;
      la::TransposeInit(kinetic_mat, &kinetic_trans);
      eri::AddSubmatrix(shells[j].matrix_indices(), shells[i].matrix_indices(),
                        kinetic_trans, &kinetic_energy_integrals);
      
      for (index_t k = 0; k < nuclear_centers_.n_cols(); k++) {
        
        // compute nuclear
        Vector c_vec;
        nuclear_centers_.MakeColumnVector(k, &c_vec);
        Vector nuclear;
        eri::ComputeNuclearIntegrals(shells[i], shells[j], 
                                     c_vec, nuclear_charges_[k], &nuclear);
        //nuclear.PrintDebug("Nuclear integrals");
        
        Matrix nuclear_mat;
        nuclear_mat.Copy(nuclear.ptr(), shells[i].num_functions(), 
                         shells[j].num_functions());
        
        eri::AddSubmatrix(shells[i].matrix_indices(), shells[j].matrix_indices(), 
                          nuclear_mat, &potential_energy_integrals);
        
        Matrix nuclear_trans;
        la::TransposeInit(nuclear_mat, &nuclear_trans);
        eri::AddSubmatrix(shells[j].matrix_indices(), shells[i].matrix_indices(), 
                          nuclear_trans, &potential_energy_integrals);
        
      } // for k 
      
    } // for j
    
    // don't forget the entries on the diagonal
    
    Vector overlap;
    eri::ComputeOverlapIntegrals(shells[i], shells[i], &overlap);
    Matrix overlap_mat;
    overlap_mat.Copy(overlap.ptr(), shells[i].num_functions(), 
                     shells[i].num_functions());
    //free(overlap);
    
    eri::AddSubmatrix(shells[i].matrix_indices(), shells[i].matrix_indices(),
                      overlap_mat, &overlap_matrix_);
    
    Vector kinetic;
    eri::ComputeKineticIntegrals(shells[i], shells[i], &kinetic);
    
    Matrix kinetic_mat;
    kinetic_mat.Copy(kinetic.ptr(), shells[i].num_functions(), 
                     shells[i].num_functions());
    //free(kinetic);
    
    eri::AddSubmatrix(shells[i].matrix_indices(), shells[i].matrix_indices(),
                      kinetic_mat, &kinetic_energy_integrals);
    
    for (index_t k = 0; k < nuclear_centers_.n_cols(); k++) {
      
      // compute nuclear
      Vector c_vec;
      nuclear_centers_.MakeColumnVector(k, &c_vec);
      Vector nuclear;
      eri::ComputeNuclearIntegrals(shells[i], shells[i], 
                                   c_vec, nuclear_charges_[k], &nuclear);
      //nuclear.PrintDebug("Nuclear integrals");
      
      Matrix nuclear_mat;
      nuclear_mat.Copy(nuclear.ptr(), shells[i].num_functions(), 
                       shells[i].num_functions());
      //free(nuclear);
      
      eri::AddSubmatrix(shells[i].matrix_indices(), shells[i].matrix_indices(), 
                        nuclear_mat, &potential_energy_integrals);
      
    } // for k      
    
  } // for i
  
  //overlap_matrix_.PrintDebug("overlap");
  //kinetic_energy_integrals.PrintDebug("kinetic");
  //potential_energy_integrals.PrintDebug("potential");
  //la::Scale(-1.0, &kinetic_energy_integrals_);
  la::Scale(-1.0, &potential_energy_integrals);
  
  la::AddInit(kinetic_energy_integrals, potential_energy_integrals, 
              &core_hamiltonian_);
  
} // ComputeCoreMatrices_()

void FockMatrixComparison::ComputeChangeOfBasisMatrix_() {
 
  Matrix left_vectors;
  Vector eigenvalues;
  Matrix right_vectors_trans;
  
  //overlap_matrix_.PrintDebug("Overlap matrix");
  
  success_t eigenval_success = la::SVDInit(overlap_matrix_, &eigenvalues, 
                                           &left_vectors, &right_vectors_trans);
  
  //eigenvalues.PrintDebug("eigenvalues");
  
  if (eigenval_success == SUCCESS_FAIL) {
    FATAL("Unable to Compute Eigenvalues of Overlap Matrix");
  }
  
  double *min_eigenval;
  min_eigenval = std::min_element(eigenvalues.ptr(), 
                                  eigenvalues.ptr() + eigenvalues.length());
  
  printf("Smallest Eigenvalue of Overlap Matrix: %g\n", *min_eigenval);
  fx_result_double(my_mod_, "smallest_overlap_eigenvalue", *min_eigenval);
  
#ifdef DEBUG
  
  //eigenvalues.PrintDebug("eigenvalues");
  
  for (index_t i = 0; i < eigenvalues.length(); i++) {
    DEBUG_ASSERT_MSG(!isnan(eigenvalues[i]), 
                     "Complex eigenvalue in diagonalizing overlap matrix.\n");
    
    if (eigenvalues[i] < 0.0) {
     
      printf("eigenvalue: %g, left_vec: %g, right_vec: %g\n", eigenvalues[i], 
             left_vectors.ref(0,i), right_vectors_trans.ref(i,0));
      
    }
    
    DEBUG_WARN_MSG_IF(eigenvalues[i] < 0.0001, 
                      "negative or near-zero eigenvalue in overlap_matrix");
    
    /*
    Vector eigenvec;
    left_vectors.MakeColumnVector(i, &eigenvec);
    double len = la::LengthEuclidean(eigenvec);
    DEBUG_APPROX_DOUBLE(len, 1.0, 0.001);
    */
    
    /*
    for (index_t j = i+1; j < eigenvalues.length(); j++) {
      
      Vector eigenvec2;
      left_vectors.MakeColumnVector(j, &eigenvec2);
      
      double dotprod = la::Dot(eigenvec, eigenvec2);
      DEBUG_APPROX_DOUBLE(dotprod, 0.0, 0.001);
      
    }
     */
  }
  
#endif
  
  for (index_t i = 0; i < eigenvalues.length(); i++) {
    DEBUG_ASSERT(eigenvalues[i] > 0.0);
    eigenvalues[i] = 1.0/sqrt(eigenvalues[i]);
  }
  
  Matrix sqrt_lambda;
  sqrt_lambda.InitDiagonal(eigenvalues);
  
  Matrix lambda_times_u_transpose;
  la::MulTransBInit(sqrt_lambda, left_vectors, &lambda_times_u_transpose);
  la::MulInit(left_vectors, lambda_times_u_transpose, 
              &change_of_basis_matrix_);
  
}

void FockMatrixComparison::DiagonalizeFockMatrix_() {
 
  la::AddInit(*G_mat_, core_hamiltonian_, &fock_mat_);
  la::AddInit(*naive_G_mat_, core_hamiltonian_, &naive_fock_mat_);
  
  Matrix comp_coefficients_prime;
  Matrix comp_right_vectors_trans;
  
  Matrix naive_coefficients_prime;
  Matrix naive_right_vectors_trans;
  
  la::SVDInit(fock_mat_, &comp_energies_, &comp_coefficients_prime, 
              &comp_right_vectors_trans);
  la::SVDInit(naive_fock_mat_, &naive_energies_, &naive_coefficients_prime, 
              &naive_right_vectors_trans);
  
  
  for (index_t i = 0; i < num_entries_; i++) {
    
    // if the left and right vector don't have equal signs the eigenvector 
    // is negative
    
    if ((comp_coefficients_prime.ref(0,i) > 0 
         && comp_right_vectors_trans.ref(i,0) < 0) 
        || (comp_coefficients_prime.ref(0,i) < 0 && 
            comp_right_vectors_trans.ref(i,0) > 0)){
      
      comp_energies_[i] = -1 * comp_energies_[i];
      
    }
    
    if ((naive_coefficients_prime.ref(0,i) > 0 
         && naive_right_vectors_trans.ref(i,0) < 0) 
        || (naive_coefficients_prime.ref(0,i) < 0 && 
            naive_right_vectors_trans.ref(i,0) > 0)){
      
      naive_energies_[i] = -1 * naive_energies_[i];
      
    }
    
  }
  
}

void FockMatrixComparison::CompareEigenvalues_() {
  
  Vector energy_diff;
  la::SubInit(comp_energies_, naive_energies_, &energy_diff);
  
  double energy_diff_l2_norm = la::Dot(energy_diff, energy_diff);
  
  fx_result_double(my_mod_, "energy_l2_norm", energy_diff_l2_norm);
  
  // compute total energies
  
  naive_total_energy_ = 0.0;
  comp_total_energy_ = 0.0;
  
  for (index_t i = 0; i < num_entries_; i++) {
    
    naive_total_energy_ += density_.ref(i, i) * 
        (core_hamiltonian_.ref(i,i) + naive_fock_mat_.ref(i,i));
    
    comp_total_energy_ += density_.ref(i,i) * 
        (core_hamiltonian_.ref(i,i) + fock_mat_.ref(i,i));
    
    for (index_t j = i; j < num_entries_; j++) {
      
      naive_total_energy_ += 2 * density_.ref(i, j) * 
          (core_hamiltonian_.ref(i,j) + naive_fock_mat_.ref(i,j));
      
      comp_total_energy_ += 2 * density_.ref(i,j) * 
          (core_hamiltonian_.ref(i,j) + fock_mat_.ref(i,j));
      
    } // for j
    
  } // for i
  
  naive_total_energy_ *= 0.5;
  comp_total_energy_ *= 0.5;
  
  fx_result_double(my_mod_, "naive_energy", naive_total_energy_);
  fx_result_double(my_mod_, "comp_energy", comp_total_energy_);
  
  double total_energy_diff = fabs(naive_total_energy_ - comp_total_energy_);
  double total_rel_energy_diff = total_energy_diff / fabs(naive_total_energy_);
  
  // note that these lack the nuclear repulsion energy
  fx_result_double(my_mod_, "energy_diff", total_energy_diff);
  fx_result_double(my_mod_, "rel_energy_diff", total_rel_energy_diff);
  
} // compare eigenvalues



void FockMatrixComparison::Init(fx_module* exp_mod, Matrix** exp_mats, 
                                fx_module* naive_mod, 
                                Matrix** naive_mats, const Matrix& centers, 
                                const Matrix& exp, const Matrix& momenta, 
                                const Matrix& density, Matrix* nuclear_centers, 
                                const Matrix& nuclear_charges, 
                                fx_module* my_mod,
                                const char* one_electron_name, 
                                const char* change_of_basis_name) {
  
  
  /*
  core_file_ = (char *)malloc(strlen(one_electron_name) * sizeof(char));
  change_file_ = (char *)malloc(strlen(change_of_basis_name) * sizeof(char));

  memcpy(core_file_, one_electron_name, strlen(one_electron_name));
  memcpy(change_file_, change_of_basis_name, strlen(change_of_basis_name));
  */
  
  core_string_ = one_electron_name;
  change_string_ = change_of_basis_name;
  
  approx_mod_ = exp_mod;
  naive_mod_ = naive_mod;
  
  G_mat_ = exp_mats[0];
  J_mat_ = exp_mats[1];
  K_mat_ = exp_mats[2];
  
  naive_G_mat_ = naive_mats[0];
  naive_J_mat_ = naive_mats[1];
  naive_K_mat_ = naive_mats[2];
  
  compare_fock_ = G_mat_ != NULL;
  compare_coulomb_ = J_mat_ != NULL;
  compare_exchange_ = K_mat_ != NULL;
    
  my_mod_ = my_mod;
  
  compare_energies_ = fx_param_bool(my_mod_, "energies", true);
  
  max_diff_F_ = -1.0;
  max_diff_J_ = -1.0;
  max_diff_K_ = -1.0;
  
  max_rel_F_ = -1.0;
  max_rel_J_ = -1.0;
  max_rel_K_ = -1.0;
  
  total_diff_F_ = 0.0;
  total_diff_J_ = 0.0;
  total_diff_K_ = 0.0;
  
  rms_F_ = 0.0;
  rms_J_ = 0.0;
  rms_K_ = 0.0;
  
  num_entries_ = naive_G_mat_->n_cols();
  
  rel_error_cutoff_ = 10e-12;
  
  centers_.Copy(centers);
  exponents_.Copy(exp.ptr(), centers_.n_cols());
  momenta_.Copy(momenta.ptr(), centers_.n_cols());
  density_.Copy(density);
  
  overlap_matrix_.Init(num_entries_, num_entries_);
  overlap_matrix_.SetZero();
  
  // if no centers specified, then can't compare energies
  if (nuclear_centers) {
    nuclear_centers_.Copy(*nuclear_centers);
    nuclear_charges_.Copy(nuclear_charges.ptr(), nuclear_centers_.n_cols());
  }
  else {
    //printf("setting compare energies to false\n");
    compare_energies_ = false;
    nuclear_charges_.Init(1);
    nuclear_centers_.Init(1,1);
  }
    
  // check that all matrices are the same size
  
} // Init()


void FockMatrixComparison::Compare() {

  printf("====== Comparing Matrices ======\n");

  for (index_t i = 0; i < num_entries_; i++) {
  
    for (index_t j = i; j < num_entries_; j++) {
    
      if (compare_fock_) {
      
        double F_entry1 = G_mat_->ref(i, j);
        double F_entry2 = naive_G_mat_->ref(i, j);
      
        double this_diff = fabs(F_entry1 - F_entry2);
        double this_rel_diff = fabs(this_diff/F_entry2);
            
        if (this_diff > max_diff_F_) {
          max_diff_F_ = this_diff;
        }
        
        // add in the relative error cutoff to keep from storing large 
        // relative errors from very small sources
        if ((this_rel_diff > max_rel_F_) && (this_rel_diff > rel_error_cutoff_)
            && (F_entry1 != 0.0)) {
          max_rel_F_ = this_rel_diff;
          max_index_row_F_ = i;
          max_index_col_F_ = j;
        }
        
        // account for symmetry
        if (i != j) {
          this_diff *= 2;
        }
        
        total_diff_F_ += this_diff;
        rms_F_ += this_diff * this_diff;
            
      } // compare_fock
      
      if (compare_coulomb_) {
      
        double J_entry1 = J_mat_->ref(i, j);
        double J_entry2 = naive_J_mat_->ref(i, j);
        
        double this_diff = fabs(J_entry1 - J_entry2);
        double this_rel_diff = fabs(this_diff/J_entry2);
        
        if (this_diff > max_diff_J_) {
          max_diff_J_ = this_diff;
          max_abs_index_row_J_ = i;
          max_abs_index_col_J_ = j;
        }
        
        if ((this_rel_diff > max_rel_J_) && (this_rel_diff > rel_error_cutoff_)
            && (J_entry1 != 0.0)) {
          max_rel_J_ = this_rel_diff;
          max_index_row_J_ = i;
          max_index_col_J_ = j;
        }
        
        // account for symmetry
        if (i != j) {
          this_diff *= 2;
        }
        
        total_diff_J_ += this_diff;  
        rms_J_ += this_diff * this_diff;
      
      } // compare coulomb
      
      if (compare_exchange_) {
        
        double K_entry1 = K_mat_->ref(i, j);
        double K_entry2 = naive_K_mat_->ref(i, j);
        
        double this_diff = fabs(K_entry1 - K_entry2);
        double this_rel_diff = fabs(this_diff/K_entry2);
        
        if (this_diff > max_diff_K_) {
          max_diff_K_ = this_diff;
        }
        
        if ((this_rel_diff > max_rel_K_) && (this_rel_diff > rel_error_cutoff_)
            && (K_entry1 != 0.0)) {
          max_rel_K_ = this_rel_diff;
          max_index_row_K_ = i;
          max_index_col_K_ = j;
        }
        
        // account for symmetry
        if (i != j) {
          this_diff *= 2;
        }
        
        total_diff_K_ += this_diff;
        rms_K_ += this_diff * this_diff;
        
      } // compare exchange
      
    } // for j
  
  } // for i
  
  
  // compute eigenvalue differences
  
  //printf("compare_energies: %s\n", compare_energies_ ? "true" : "false");
  //printf("compare_fock: %s\n", compare_fock_ ? "true" : "false");

  
  if (compare_energies_ && compare_fock_) {
    
    printf("=== Comparing Spectra ===\n");
    
    
    const char* core_file = core_string_.c_str();
    const char* change_file = change_string_.c_str();
    // compute H and S
    // this will require the centers and exponents and momenta and nuclear 
    // centers
    
    bool matrices_loaded = true;
    if (data::Load(core_file, &core_hamiltonian_) == SUCCESS_FAIL) {
      core_hamiltonian_.Destruct();
      matrices_loaded = false;
    }
    if (matrices_loaded && 
        (data::Load(change_file, &change_of_basis_matrix_) == SUCCESS_FAIL)) {
      change_of_basis_matrix_.Destruct();
      matrices_loaded = false;
    }
    
    if (!matrices_loaded) {
      ComputeCoreMatrices_();
      
      ComputeChangeOfBasisMatrix_();
      
      // save the matrices to files
      
      data::Save(core_file, core_hamiltonian_);
      data::Save(change_file, change_of_basis_matrix_);
    }
    else {
      // to keep it from complaining at the end
      //overlap_matrix_.Init(1,1); 
    }
        
    // compute eigenvalues
    
    DiagonalizeFockMatrix_();
    
    // compare them
    
    CompareEigenvalues_();
    
  } // compare fock eigenvalues
  else {
    // need to initialize the matrices to prevent a crash
    
    core_hamiltonian_.Init(1,1);
    //overlap_matrix_.Init(1,1);
    change_of_basis_matrix_.Init(1,1);
    fock_mat_.Init(1,1);
    naive_fock_mat_.Init(1,1);
    comp_energies_.Init(1);
    naive_energies_.Init(1);
    
  }
  
  // output results 
/*  
  printf("exact: %g, approx: %g\n", naive_J_mat_->ref(max_abs_index_row_J_, max_abs_index_col_J_), 
         J_mat_->ref(max_abs_index_row_J_, max_abs_index_col_J_));
  */
  if (compare_fock_) {
    fx_result_double(my_mod_, "max_diff_F", max_diff_F_);
    //fx_result_int(my_mod_, "max_index_row_F", max_index_row_F_);
    //fx_result_int(my_mod_, "max_index_col_F", max_index_col_F_);
    //fx_result_double(my_mod_, "ave_diff_F", 
    //                 (total_diff_F_/(num_entries_ * num_entries_)));
    fx_result_double(my_mod_, "max_rel_diff_F", max_rel_F_);
    fx_result_double(my_mod_, "rms_F", sqrt(rms_F_ / (double)(num_entries_ * num_entries_)));
  }

  if (compare_coulomb_) {
    fx_result_double(my_mod_, "max_diff_J", max_diff_J_);
    //fx_result_int(my_mod_, "max_abs_index_row_J", max_abs_index_row_J_);
    //fx_result_int(my_mod_, "max_abs_index_col_J", max_abs_index_col_J_);
    
    fx_result_double(my_mod_, "max_rel_diff_J", max_rel_J_);
    //fx_result_int(my_mod_, "max_index_row_J", max_index_row_J_);
    //fx_result_int(my_mod_, "max_index_col_J", max_index_col_J_);
    //fx_result_double(my_mod_, "ave_diff_J", 
    //                 (total_diff_J_/(num_entries_ * num_entries_)));
    fx_result_double(my_mod_, "rms_J", sqrt(rms_J_ / (double)(num_entries_ * num_entries_)));
    
  }

  if (compare_exchange_) {
    fx_result_double(my_mod_, "max_diff_K", max_diff_K_);
    //fx_result_int(my_mod_, "max_index_row_K", max_index_row_K_);
    //fx_result_int(my_mod_, "max_index_col_K", max_index_col_K_);
    //fx_result_double(my_mod_, "ave_diff_K", 
    //                 (total_diff_K_/(num_entries_ * num_entries_)));
    fx_result_double(my_mod_, "max_rel_diff_K", max_rel_K_);
    fx_result_double(my_mod_, "rms_K", sqrt(rms_K_ / (double)(num_entries_ * num_entries_)));
  }
                   
} // Compare()

// not sure this class actually owns these matrices
void FockMatrixComparison::Destruct() {

  if (G_mat_) {
    G_mat_->Destruct();
  }
  if (naive_G_mat_) {
    naive_G_mat_->Destruct();
  }
  if (J_mat_) {
    J_mat_->Destruct();
  }
  if (naive_J_mat_) {
    naive_J_mat_->Destruct();
  }
  if (K_mat_) {
    K_mat_->Destruct();
  }
  if (naive_K_mat_) {
    naive_K_mat_->Destruct();
  }
  
  my_mod_ = NULL;
  approx_mod_ = NULL;
  naive_mod_ = NULL;
  
  
} // Destruct()




