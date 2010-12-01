#include "scf_solver.h"

/*
template <class CoulombAlg, class ExchangeAlg>
void SCFSolver::FormChangeOfBasisMatrix_() {
  
  Matrix left_vectors;
  Vector eigenvalues;
  Matrix right_vectors_trans;
  
  la::SVDInit(overlap_matrix_, &eigenvalues, &left_vectors, 
              &right_vectors_trans);
  
#ifdef DEBUG
  
  eigenvalues.PrintDebug();
  
  for (index_t i = 0; i < eigenvalues.length(); i++) {
    DEBUG_ASSERT_MSG(!isnan(eigenvalues[i]), 
                     "Complex eigenvalue in diagonalizing overlap matrix.\n");
    
    DEBUG_WARN_MSG_IF(fabs(eigenvalues[i]) < 0.001, 
                      "near-zero eigenvalue in overlap_matrix");
    
    Vector eigenvec;
    left_vectors.MakeColumnVector(i, &eigenvec);
    double len = la::LengthEuclidean(eigenvec);
    DEBUG_APPROX_DOUBLE(len, 1.0, 0.001);
    
    for (index_t j = i+1; j < eigenvalues.length(); j++) {
      
      Vector eigenvec2;
      left_vectors.MakeColumnVector(j, &eigenvec2);
      
      double dotprod = la::Dot(eigenvec, eigenvec2);
      DEBUG_APPROX_DOUBLE(dotprod, 0.0, 0.001);
      
    }
  }
  
#endif
  
  for (index_t i = 0; i < eigenvalues.length(); i++) {
    DEBUG_ASSERT(eigenvalues[i] > 0.0);
    eigenvalues[i] = 1/sqrt(eigenvalues[i]);
  }
  
  Matrix sqrt_lambda;
  sqrt_lambda.InitDiagonal(eigenvalues);
  
  Matrix lambda_times_u_transpose;
  la::MulTransBInit(sqrt_lambda, left_vectors, &lambda_times_u_transpose);
  la::MulInit(left_vectors, lambda_times_u_transpose, 
              &change_of_basis_matrix_);
  
  printf("Change Of Basis Matrix:\n");
  change_of_basis_matrix_.PrintDebug();
  
  
} // FormChangeOfBasisMatrix_()
*/

/*
template <class CoulombAlg, class ExchangeAlg>
void SCFSolver::ComputeDensityMatrix_() {
  
  FillOrbitals_();
  
  
   ot::Print(occupied_indices_);
   energy_vector_.PrintDebug();
   
  
  density_matrix_frobenius_norm_ = 0.0;
  
  // Rows of density_matrix
  for (index_t density_row = 0; density_row < number_of_basis_functions_;
       density_row++) {
    
    // Columns of density matrix
    for (index_t density_column = 0; 
         density_column < number_of_basis_functions_; density_column++) {
      
      // Occupied orbitals
      double this_sum = 0.0;
      for (index_t occupied_index = 0; 
           occupied_index < number_to_fill_; occupied_index++) {
        
        this_sum = this_sum + (
                               coefficient_matrix_.ref(
                                                       density_row, occupied_indices_[occupied_index]) * 
                               coefficient_matrix_.ref(
                                                       density_column, occupied_indices_[occupied_index]));
        
      } // occupied_index
      
      double this_entry = density_matrix_.ref(density_row, density_column);
      
      // Leach says there is a factor of 2 here
      this_sum = 2 * this_sum;
      
      double this_diff = this_sum - this_entry;
      
      // Computing the frobenius norm of the difference between this 
      // iteration's density matrix and the previous one for testing 
      // convergence
      
      density_matrix_frobenius_norm_ += (this_diff * this_diff); 
      
      
      density_matrix_.set(density_row, density_column, this_sum);
      
    } // density_column
    
  } //density_row
  
  iteration_density_norms_[current_iteration_] = 
    density_matrix_frobenius_norm_;
  
} // ComputeDensityMatrix_
*/

/*
template <class CoulombAlg, class ExchangeAlg>
void SCFSolver::ComputeDensityMatrixDIIS_() {
  
  FillOrbitals_();
  
  //density_matrix_norms_.SetZero();
  
  // Rows of density_matrix
  for (index_t density_row = 0; density_row < number_of_basis_functions_;
       density_row++) {
    
    // Columns of density matrix
    for (index_t density_column = 0; 
         density_column < number_of_basis_functions_; density_column++) {
      
      // Occupied orbitals
      double this_sum = 0.0;
      
      for (index_t occupied_index = 0; 
           occupied_index < number_to_fill_; occupied_index++) {
        
        this_sum = this_sum + (coefficient_matrix_.ref(density_row, 
                                                       occupied_indices_[occupied_index]) * 
                               coefficient_matrix_.ref(density_column, 
                                                       occupied_indices_[occupied_index]));
        
      } // occupied_index
      
      this_sum = 2 * this_sum;
      
      density_matrices_[diis_index_].set(density_row, density_column, 
                                         this_sum);
      
      // find the difference between this matrix and last iterations soln.
      double this_error = this_sum - 
        density_matrix_.ref(density_row, density_column);
      
      density_matrix_errors_[diis_index_].set(density_row, density_column, 
                                              this_error);
      
    } // density_column
    
  } //density_row
  
  const double* err_ptr = density_matrix_errors_[diis_index_].ptr();
  
  index_t len = number_of_basis_functions_ * number_of_basis_functions_;
  
  for (index_t i = 0; i < diis_count_; i++) {
    
    const double* this_err_ptr = density_matrix_errors_[i].ptr();
    
    double this_norm = la::Dot(len, err_ptr, this_err_ptr);
    
    density_matrix_norms_.set(diis_index_, i, this_norm);
    density_matrix_norms_.set(i, diis_index_, this_norm);
    
    
     density_matrix_norms_.set(diis_count_, i, -1);
     density_matrix_norms_.set(i, diis_count_, -1);
     
    
  }
  
  printf("diis_index: %d\n", diis_index_);
  density_matrix_norms_.PrintDebug();
  
  DIISSolver_();
  
  diis_index_++;
  diis_index_ = diis_index_ % diis_count_;
  
} // ComputeDensityMatrixDIIS_()
*/

/*
template <class CoulombAlg, class ExchangeAlg>
void SCFSolver::DIISSolver_() {
  
  Matrix old_density;
  old_density.Copy(density_matrix_);
  
  // Make this plus one, since the first entry doesn't mean much
  if (likely(current_iteration_ > diis_count_ + 1)) {
    
    Vector diis_coeffs;
    
    la::SolveInit(density_matrix_norms_, diis_rhs_, &diis_coeffs);
    
    density_matrix_.SetZero();
    
    for (index_t i = 0; i < diis_count_; i++) {
      
      // Should scale density_matrices_[i] by the right value and add to 
      // the overall density matrix
      la::AddExpert(diis_coeffs[i], density_matrices_[i], &density_matrix_);
      
    }
    
    diis_coeffs.PrintDebug();
    
  }
  else {
    
    
    density_matrix_.CopyValues(density_matrices_[diis_index_]);
    
    
  }
  
  la::SubFrom(density_matrix_, &old_density);
  
  density_matrix_frobenius_norm_ = la::Dot(number_of_basis_functions_ * 
                                           number_of_basis_functions_, 
                                           old_density.ptr(), 
                                           old_density.ptr());
  
  
  iteration_density_norms_[current_iteration_] = 
    density_matrix_frobenius_norm_;    
  
} // DIISSolver_()
*/

/*
template <class CoulombAlg, class ExchangeAlg>
void SCFSolver::DiagonalizeFockMatrix_() {
  
  energy_vector_.Destruct();
  Matrix coefficients_prime;
  Matrix right_vectors_trans;
  
  //la::EigenvectorsInit(fock_matrix_, &energy_vector_, &coefficients_prime);
  
  la::SVDInit(fock_matrix_, &energy_vector_, &coefficients_prime, 
              &right_vectors_trans);
  
  
  for (index_t i = 0; i < number_of_basis_functions_; i++) {
    
    // if the left and right vector don't have equal signs the eigenvector 
    // is negative
    
    if ((coefficients_prime.ref(0,i) > 0 && right_vectors_trans.ref(i,0) < 0) 
        || (coefficients_prime.ref(0,i) < 0 && 
            right_vectors_trans.ref(i,0) > 0)){
      
      energy_vector_[i] = -1 * energy_vector_[i];
      
    }
    
  }
  
  
  
   printf("Fock matrix:\n");
   fock_matrix_.PrintDebug();
   
   printf("Right vector:\n");
   right_vectors_trans.PrintDebug();
   
   printf("Energies:\n");
   energy_vector_.PrintDebug();
   
   printf("Coefficients (prime):\n");
   coefficients_prime.PrintDebug();
   
  
#ifdef DEBUG
  
  for (index_t i = 0; i < energy_vector_.length(); i++) {
    DEBUG_ASSERT_MSG(!isnan(energy_vector_[i]), 
                     "Complex eigenvalue in diagonalizing Fock matrix.\n");
  }
  
#endif
  
  // 3. Find the untransformed eigenvector matrix
  la::MulOverwrite(change_of_basis_matrix_, coefficients_prime, 
                   &coefficient_matrix_);
  
} // DiagonalizeFockMatrix_
*/

/*
template <class CoulombAlg, class ExchangeAlg>
void SCFSolver::ComputeOneElectronMatrices_() {
  
  for (index_t row_index = 0; row_index < number_of_basis_functions_; 
       row_index++) {
    
    for (index_t col_index = row_index; col_index < number_of_basis_functions_; 
         col_index++) {
      
      Vector row_vec;
      basis_centers_.MakeColumnVector(row_index, &row_vec);
      Vector col_vec;
      basis_centers_.MakeColumnVector(col_index, &col_vec);
      double dist = la::DistanceSqEuclidean(row_vec, col_vec);
      
      double kinetic_integral = ComputeKineticIntegral_(dist);
      
      double overlap_integral = ComputeOverlapIntegral_(dist);
      
      double nuclear_integral = 0.0;
      for (index_t nuclear_index = 0; nuclear_index < number_of_nuclei_; 
           nuclear_index++) {
        
        Vector nuclear_position;
        nuclear_centers_.MakeColumnVector(nuclear_index, &nuclear_position);
        
        nuclear_integral = nuclear_integral + 
          ComputeNuclearIntegral_(nuclear_position, nuclear_index, row_vec, 
                                  col_vec);
        
      } // nuclear_index
      
      kinetic_energy_integrals_.set(row_index, col_index, kinetic_integral);
      potential_energy_integrals_.set(row_index, col_index, nuclear_integral);
      overlap_matrix_.set(row_index, col_index, overlap_integral);
      
      if (likely(row_index != col_index)) {
        kinetic_energy_integrals_.set(col_index, row_index, kinetic_integral);
        potential_energy_integrals_.set(col_index, row_index, nuclear_integral);
        overlap_matrix_.set(col_index, row_index, overlap_integral);
      }
      
    } // column_index
    
  } // row_index
  
  la::AddInit(kinetic_energy_integrals_, potential_energy_integrals_, 
              &core_matrix_);
  
} // ComputeOneElectronMatrices_()
*/

/*
template <class CoulombAlg, class ExchangeAlg>
double SCFSolver::ComputeNuclearRepulsion_() {
  
  double nuclear_energy = 0.0;
  
  for (index_t a = 0; a < number_of_nuclei_; a++) {
    
    for (index_t b = a+1; b < number_of_nuclei_; b++) {
      
      Vector a_vec; 
      nuclear_centers_.MakeColumnVector(a, &a_vec);
      Vector b_vec;
      nuclear_centers_.MakeColumnVector(b, &b_vec);
      
      double dist_sq = la::DistanceSqEuclidean(a_vec, b_vec);
      double dist = sqrt(dist_sq);
      
      nuclear_energy += nuclear_masses_[a] * nuclear_masses_[b] / dist;
      
    } // b
    
  } // a
  
  return nuclear_energy;
  
} // ComputeNuclearRepulsion_()
*/

