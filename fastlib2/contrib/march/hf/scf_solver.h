/**
 * @file scf_solver.h
 * 
 * @author Bill March (march@gatech.edu)
 *
 * Contains an algorithm class for the SCF solution for Hartree-Fock.  
 */

#ifndef SCF_SOLVER_H
#define SCF_SOLVER_H

#include <fastlib/fastlib.h>
#include "dual_tree_integrals.h"
#include "naive_fock_matrix.h"

/**
 * Algorithm class for the SCF part of the HF computation.  This class assumes 
 * the integrals have been computed and does the SVD-like part of the 
 * computation.
 */
class SCFSolver {
  
  friend class SCFSolverTest;
  
  FORBID_ACCIDENTAL_COPIES(SCFSolver);
  
 private:
  
  Matrix basis_centers_;
  Matrix nuclear_centers_;  
  
  Vector nuclear_masses_;
  
  Matrix core_matrix_; // T + V
  Matrix kinetic_energy_integrals_; // T
  Matrix potential_energy_integrals_; // V
  
  Matrix coefficient_matrix_; // C or C'
  
  // Consider changing this name to reflect that it's the change of basis matrix
  Matrix overlap_matrix_; // S 
  Matrix change_of_basis_matrix_; // S^{-1/2} 
  Matrix density_matrix_; // D
  Matrix fock_matrix_; // F or F', depending on the basis
  
  Vector energy_vector_; // The diagonal matrix of eigenvalues of F/F'
  
  index_t number_of_basis_functions_; // N
  index_t number_of_electrons_; // K
  index_t number_of_nuclei_; 
  index_t number_to_fill_;
  
  // I think I'll have to compute this in the beginning
  double nuclear_repulsion_energy_;
  
  double one_electron_energy_;
  double two_electron_energy_;
  
  ArrayList<double> total_energy_;
  index_t current_iteration_;
  ArrayList<double> density_matrix_norms_;
  
  
  static const index_t expected_number_of_iterations_ = 40;
  
  double density_convergence_;
  double energy_convergence_;
  
  // Used for computing the convergence of the density matrix on the fly
  double density_matrix_frobenius_norm_;
  
  struct datanode* module_;
  
  DualTreeIntegrals integrals_;
  
  NaiveFockMatrix naive_integrals_;
  
  bool do_naive_;
  
  ArrayList<index_t> occupied_indices_;
  
  ArrayList<index_t> old_from_new_centers_;
  
  double bandwidth_;
  
  double normalization_constant_squared_;
  
  // Track the largest and smallest entries in the fock matrix in order to
  // understand pruning
  double fock_max_;
  double fock_min_;
  double density_max_;
  double density_min_;
  double coulomb_max_;
  double coulomb_min_;
  double exchange_max_;
  double exchange_min_;
  
 public:
    
  SCFSolver() {}
  
  ~SCFSolver() {}
  

  void Init(struct datanode* mod, index_t num_electrons, 
            const Matrix& basis_centers, const Matrix& density, 
            const Matrix& nuclear, const Vector& nuclear_mass) {
    
    module_ = mod;
    number_of_electrons_ = num_electrons;
    
    struct datanode* integral_mod = fx_submodule(module_, "integrals", 
                                                "integrals");
                                                
    struct datanode* naive_mod = fx_submodule(module_, "naive_integrals", 
                                              "naive_integrals");
    
    bandwidth_ = fx_param_double(module_, "bandwidth", 0.1);
    
    do_naive_ = fx_param_exists(NULL, "naive");
    
    if (do_naive_) {
    
      naive_integrals_.Init(basis_centers, naive_mod, density, bandwidth_);
      basis_centers_.Copy(basis_centers);
      density_matrix_.Copy(density);
    
    }
    else {
    
      integrals_.Init(basis_centers, integral_mod, bandwidth_);
      
      // Need to get out the permutation from the integrals_, then use it to 
      // permute the basis centers
      
      integrals_.GetPermutation(&old_from_new_centers_);
      //ot::Print(old_from_new_centers_);
      
      PermuteMatrix_(basis_centers, &basis_centers_, old_from_new_centers_);
      /*
      printf("Old basis:\n");
      basis_centers.PrintDebug();
      printf("New basis:\n");
      basis_centers_.PrintDebug();
      */
      PermuteMatrix_(density, &density_matrix_, old_from_new_centers_);
      
      integrals_.SetDensity(density_matrix_);
      
    }
    
    nuclear_centers_.Copy(nuclear);
    
    nuclear_masses_.Copy(nuclear_mass);
    
    number_of_nuclei_ = nuclear_centers_.n_cols();
    
    number_to_fill_ = (index_t)ceil((double)number_of_electrons_/2);
    occupied_indices_.Init(number_to_fill_);
    
    DEBUG_ASSERT(number_of_nuclei_ == nuclear_masses_.length());
    
    
    number_of_basis_functions_ = basis_centers_.n_cols();
    
    DEBUG_ASSERT(number_of_basis_functions_ >= number_to_fill_);
    
    // Empty inits to prevent errors on closing
    overlap_matrix_.Init(number_of_basis_functions_, 
                         number_of_basis_functions_);
    kinetic_energy_integrals_.Init(number_of_basis_functions_, 
                                   number_of_basis_functions_);
    potential_energy_integrals_.Init(number_of_basis_functions_, 
                                     number_of_basis_functions_);
    
    coefficient_matrix_.Init(number_of_basis_functions_, 
                             number_of_basis_functions_);
    
    energy_vector_.Init(number_of_basis_functions_);
    
    total_energy_.Init(expected_number_of_iterations_);
    density_matrix_norms_.Init(expected_number_of_iterations_);
    
    density_convergence_ = fx_param_double(module_, "density_convergence", 0.1);
    energy_convergence_ = fx_param_double(module_, "energy_convergence", 0.1);
    
    // Need to double check that this is right
    density_matrix_frobenius_norm_ = DBL_MAX;
    
    current_iteration_ = 0;
    
    normalization_constant_squared_ = pow((2 * bandwidth_)/math::PI, 1.5);
    
    fx_format_result(module_, "normalization", "%g", 
                     normalization_constant_squared_);
    
  } // Init()
  
 private:
 
  double ComputeOverlapIntegral_(double dist);
  
  double ComputeKineticIntegral_(double dist);
  
  double ComputeNuclearIntegral_(const Vector& nuclear_position, 
                                 index_t nuclear_index, 
                                 const Vector& mu, const Vector& nu);
    
  /**
   * Permutes the matrix mat according to the permutation given.  The permuted 
   * matrix is written to the uninitialized matrix new_mat
   */                               
  void PermuteMatrix_(const Matrix& old_mat, Matrix* new_mat, 
                      const ArrayList<index_t>& perm) {
  
    index_t num_cols = old_mat.n_cols();
    DEBUG_ASSERT(num_cols == perm.size());
    
    new_mat->Init(old_mat.n_rows(), num_cols);
    
    for (index_t i = 0; i < num_cols; i++) {
      
      Vector old_vec;
      old_mat.MakeColumnVector(perm[i], &old_vec);
      Vector new_vec;
      new_mat->MakeColumnVector(i, &new_vec);
      
      new_vec.CopyValues(old_vec);
      
    }
  
  } // PermuteMatrix_()
  
  /** 
   * Given the basis set and nuclear coordinates, compute and store the one 
   * electron matrices. 
   *
   * For now, just using loops.  In the future, it's an N-body problem but 
   * probably a very small fraction of the total running time.  
   */
  void ComputeOneElectronMatrices_();
 
 
  /**
   * Create the matrix S^{-1/2} using the eigenvector decomposition.     *
   */
  void FormChangeOfBasisMatrix_() {
/*    
    Vector eigenvalues;
    Matrix eigenvectors;
    la::EigenvectorsInit(overlap_matrix_, &eigenvalues, &eigenvectors);
    
    */
    
    Matrix left_vectors;
    Vector eigenvalues;
    Matrix right_vectors_trans;
    
    la::SVDInit(overlap_matrix_, &eigenvalues, &left_vectors, 
                &right_vectors_trans);
    
#ifdef DEBUG
    
    for (index_t i = 0; i < eigenvalues.length(); i++) {
      DEBUG_ASSERT_MSG(!isnan(eigenvalues[i]), 
                       "Complex eigenvalue in diagonalizing overlap matrix.\n");
                       
      Vector eigenvec;
      left_vectors.MakeColumnVector(i, &eigenvec);
      double len = la::LengthEuclidean(eigenvec);
      DEBUG_APPROX_DOUBLE(len, 1.0, 0.01);
      
      for (index_t j = i+1; j < eigenvalues.length(); j++) {
        
        Vector eigenvec2;
        left_vectors.MakeColumnVector(j, &eigenvec2);
        
        double dotprod = la::Dot(eigenvec, eigenvec2);
        DEBUG_APPROX_DOUBLE(dotprod, 0.0, 0.01);
        
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
    
                   
    
  } // FormChangeOfBasisMatrix_()
  
  
  /**
   * Compute the density matrix.
   * 
   * TODO: Consider an SVD or some eigenvalue solver that will find the 
   * eigenvalues in ascending order.
   */
  void ComputeDensityMatrix_() {
    
    FillOrbitals_();
    
    /*
    ot::Print(occupied_indices_);
    energy_vector_.PrintDebug();
    */
    
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
    
    density_matrix_norms_[current_iteration_] = density_matrix_frobenius_norm_;
    
  } // ComputeDensityMatrix_
  
  /**
    * Given that the Fock matrix has been transformed to the orthonormal basis
   * (F'), this function determines the energy_vector e and the transformed 
   * coefficient matrix C'.  It then untransforms the matrix to get C.
   */
  void DiagonalizeFockMatrix_() {
    
    energy_vector_.Destruct();
    
    Matrix coefficients_prime;
    
    
    Matrix right_vectors_trans;
    
    
    //la::EigenvectorsInit(fock_matrix_, &energy_vector_, &coefficients_prime);
    
    
    la::SVDInit(fock_matrix_, &energy_vector_, &coefficients_prime, 
                &right_vectors_trans);
                
#ifdef DEBUG

    Matrix right_vectors;
    la::TransposeInit(right_vectors_trans, &right_vectors);
    
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
    
      Vector i_vec;
      coefficients_prime.MakeColumnVector(i, &i_vec);
    
      for (index_t j = 0; j < number_of_basis_functions_; j++) {
        
        DEBUG_APPROX_DOUBLE(fabs(coefficients_prime.ref(i,j)), 
                            fabs(right_vectors.ref(i,j)), 0.001);
                            
        Vector j_vec;
        right_vectors.MakeColumnVector(j, &j_vec);
        DEBUG_APPROX_DOUBLE(fabs(la::Dot(i_vec, j_vec)), (i == j), 0.001);
        
      } // j
    
    } // i

#endif
    
    
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
    
      // if the left and right vector don't have equal signs the eigenvector 
      // is negative
      
      if ((coefficients_prime.ref(0,i) > 0 && right_vectors_trans.ref(i,0) < 0) 
          || (coefficients_prime.ref(0,i) < 0 && 
              right_vectors_trans.ref(i,0) > 0)){
              
        energy_vector_[i] = -1 * energy_vector_[i];
      
      }
    
    }
    
    
    /*
    printf("Fock matrix:\n");
    fock_matrix_.PrintDebug();
    
    printf("Right vector:\n");
    right_vectors_trans.PrintDebug();
    
    printf("Energies:\n");
    energy_vector_.PrintDebug();
    
    printf("Coefficients (prime):\n");
    coefficients_prime.PrintDebug();
    */
    
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
  
  
  /**
   * Determine the K/2 lowest energy orbitals.  
   * 
   * TODO: If K is odd, then the last entry here is the orbital that should 
   * have one electron.  I think the closed-shell RHF formulation I'm using
   * forbids an odd number of electons.
   *
   * I should use the built in C++ iterator-driven routines.  Ryan suggested
   * that I write iterators for ArrayLists, since that would open up a lot of
   * functionality, including sorting. 
   *
   * Now that I'm using SVD, the eigenvalues are in order up to signs.  So, I 
   * should be able to use that info to make this code more efficient.
   */
  void FillOrbitals_() {
    
    double max_energy_kept = -DBL_INF;
    index_t next_to_go = 0;
    
    for (index_t i = 0; i < number_to_fill_; i++) {
      
      occupied_indices_[i] = i;
      if (energy_vector_[i] > max_energy_kept) {
        max_energy_kept = energy_vector_[i];
        next_to_go = i;
      }
      
    }
    
    
    for (index_t i = number_to_fill_; i < number_of_basis_functions_; i++) {    
      
      double this_energy = energy_vector_[i];
      if (this_energy < max_energy_kept) {
        occupied_indices_[next_to_go] = i;
        
        // Find the new index to throw out
        double new_max = -DBL_INF;
        next_to_go = -1;
        for (index_t j = 0; j < number_to_fill_; j++) {
          if (energy_vector_[occupied_indices_[j]] > new_max) {
            new_max = energy_vector_[occupied_indices_[j]];
            next_to_go = j;
          }
        }
        max_energy_kept = new_max;
        DEBUG_ASSERT(!isinf(max_energy_kept));
        DEBUG_ASSERT(next_to_go >= 0);
        DEBUG_ASSERT(next_to_go < number_to_fill_);
      }
      
    }
    
  } //FillOrbitals_
  
  
    
  /**
   * Find the energy of the electrons in the ground state of the current 
   * wavefunction.  
   *
   * The sum needs to be over occupied orbitals, according to Szabo
   * I'm not sure about this, though, since the notation keeps changing
   */
  double ComputeElectronicEnergy_() {
    
    
    double total_energy = 0.0;
    one_electron_energy_ = 0.0;
    two_electron_energy_ = 0.0;
    
    fock_max_ = -DBL_INF;
    fock_min_ = DBL_INF;
    density_max_ = -DBL_INF;
    density_min_ = DBL_INF;
    
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
      
      // for the diagonal entries
      one_electron_energy_ += density_matrix_.ref(i,i) * core_matrix_.ref(i,i);
      two_electron_energy_ += density_matrix_.ref(i,i) * 
                              (fock_matrix_.ref(i,i) - core_matrix_.ref(i,i));
                              
      
      
      total_energy += density_matrix_.ref(i,i) * 
          (core_matrix_.ref(i,i) + fock_matrix_.ref(i,i));
          
      if (fock_matrix_.ref(i, i) > fock_max_) {
        fock_max_ = fock_matrix_.ref(i,i);
      }
      if (fock_matrix_.ref(i,i) < fock_min_) {
        fock_min_ = fock_matrix_.ref(i,i);
      } 
      if (density_matrix_.ref(i, i) > density_max_) {
        density_max_ = density_matrix_.ref(i,i);
      }
      if (density_matrix_.ref(i,i) < density_min_) {
        density_min_ = density_matrix_.ref(i,i);
      } 
      
      
      for (index_t j = i+1; j < number_of_basis_functions_; j++) {
      
        // multiply by 2 to get the lower triangle
        one_electron_energy_ += 2 * density_matrix_.ref(i,j) * 
                                core_matrix_.ref(i,j);
        two_electron_energy_ += 2 * density_matrix_.ref(i,j) * 
                                (fock_matrix_.ref(i,j) - core_matrix_.ref(i,j));
        
        double this_energy = 2 * density_matrix_.ref(i, j) * 
            (core_matrix_.ref(i, j) + fock_matrix_.ref(i, j));
            
        total_energy = total_energy + this_energy;

        if (fock_matrix_.ref(i, j) > fock_max_) {
          fock_max_ = fock_matrix_.ref(i,j);
        }
        if (fock_matrix_.ref(i,j) < fock_min_) {
          fock_min_ = fock_matrix_.ref(i,j);
        } 
        if (density_matrix_.ref(i, j) > density_max_) {
          density_max_ = density_matrix_.ref(i,j);
        }
        if (density_matrix_.ref(i,j) < density_min_) {
          density_min_ = density_matrix_.ref(i,j);
        } 
        
      } // j
    
    } // i
    
    // Leach says there is a factor of 1/2
    total_energy = (0.5 * total_energy) + nuclear_repulsion_energy_;
    
    // No factor of two because there's no overcounting of electrons
    one_electron_energy_ = one_electron_energy_;
    two_electron_energy_ = 0.5 * two_electron_energy_;
    
    printf("one_electron_energy: %g\n", one_electron_energy_);
    printf("two_electron_energy: %g\n", two_electron_energy_);
    
    return total_energy;
    
    
  } // ComputeElectronicEnergy_
  
  /**
    * Determine if the density matrix and total energy have converged.   
   */
  bool TestConvergence_() {
    
    if (unlikely(current_iteration_ < 2)) {
      return false;
    }
    
    double energy_diff = fabs(total_energy_[current_iteration_] - 
                              total_energy_[(current_iteration_ - 1)]);
    
   
    printf("current_iteration: %d\n", current_iteration_);
    
    printf("density_matrix_frobenius_norm_: %g\n", 
           density_matrix_frobenius_norm_);
           
    
    printf("total_energy_[%d]: %g\n", current_iteration_, 
           total_energy_[current_iteration_]);
    printf("energy_diff: %g\n", energy_diff);
    
    
    
    if (likely(density_matrix_frobenius_norm_ > density_convergence_)) {
      return false;
    }
    
    //printf("energy_convergence: %g\n", energy_convergence_);
    if (likely(energy_diff > energy_convergence_)) {
      return false;
    }
    
    //density_matrix_frobenius_norm_ = 0.0;
    
    return true;
    
  } // TestConvergence_
  
  /**
   * Pre- and post-multiply the Fock matrix by the change of basis matrix 
   */
  void TransformFockBasis_() {
    
    Matrix orthogonal_transpose_times_fock;
    la::MulTransAInit(change_of_basis_matrix_, fock_matrix_, 
                           &orthogonal_transpose_times_fock);
    la::MulOverwrite(orthogonal_transpose_times_fock, 
                     change_of_basis_matrix_, &fock_matrix_);
    
  } // TransformFockBasis_
  
  void UpdateFockMatrix_() {

    if (do_naive_) {
    
      naive_integrals_.UpdateMatrices(density_matrix_);
      
      naive_integrals_.ComputeFockMatrix();
      
      Matrix new_fock;
      
      naive_integrals_.PrintFockMatrix(&new_fock, NULL, NULL);
      
      la::AddOverwrite(core_matrix_, new_fock, &fock_matrix_);
    
    }
    else {
    
      integrals_.UpdateMatrices(density_matrix_);
    
      integrals_.ComputeFockMatrix();
    
      la::AddOverwrite(core_matrix_, integrals_.FockMatrix(), &fock_matrix_);
    
    }

  }
  
  /**
    * Does the SCF iterations to find the HF wavefunction
   */
  void FindSCFSolution_() {
    
    bool converged = false;
    
    /*
    printf("Matrix permutation:\n");
    ot::Print(old_from_new_centers_);
    */
    
    while (!converged) {
      
      // Step 4a.
      UpdateFockMatrix_();
      
      /*
      printf("Density Matrix\n");
      density_matrix_.PrintDebug();
      
      printf("Fock Matrix\n");
      fock_matrix_.PrintDebug();
      
      printf("Core Matrix\n");
      core_matrix_.PrintDebug();
      */
      
      // Step 4b.
      if (unlikely(current_iteration_ >= total_energy_.size())) {
        total_energy_.EnsureSizeAtLeast(2*total_energy_.size());
      }
      
      total_energy_[current_iteration_] = ComputeElectronicEnergy_();
      
      // Step 4c.
      TransformFockBasis_();
      
      // Step 4d/e.
      DiagonalizeFockMatrix_();
      
      //Step 4f.
      ComputeDensityMatrix_();
      
      // Step 4g.
      converged = TestConvergence_();
      
      current_iteration_++;
      
    } // end while
    
    total_energy_[current_iteration_] = ComputeElectronicEnergy_();
    
  } // FindSCFSolution_
  
  /**
   * Returns the nuclear repulsion energy for the nuclei given in 
   * nuclear_centers_ and nuclear_masses_
   *
   * I'm only counting each pair once, which I think is correct.  
   */
  double ComputeNuclearRepulsion_();
  
  /**
   * Sets up the matrices for the SCF iterations 
   */
  void Setup_() {
    
    nuclear_repulsion_energy_ = ComputeNuclearRepulsion_();
    
    ComputeOneElectronMatrices_();
    
    //printf("S\n");
    //overlap_matrix_.PrintDebug();
    
    FormChangeOfBasisMatrix_();
    
    fock_matrix_.Copy(core_matrix_);
    
    TransformFockBasis_();
    
    DiagonalizeFockMatrix_();
    
    ComputeDensityMatrix_();
    
    //data::Save("27_H_initial_density.csv", density_matrix_);
    
    double this_energy = ComputeElectronicEnergy_();
    
  } //Setup_

  /**
   * Save the coefficient matrix, total energy, and energy vector to files.
   *
   * TODO: think about a better way to separate the filled and virtual orbitals
   */
  void OutputResults_() {
    
    const char* coefficients_file = 
        fx_param_str(module_, "C", "coefficients.csv");
    data::Save(coefficients_file, coefficient_matrix_);
    
    const char* energy_file = fx_param_str(module_, "Etot", "total_energy.csv");
    FILE* energy_pointer = fopen(energy_file, "w");
    for (index_t i = 0; i < current_iteration_; i++) {
     
      fprintf(energy_pointer, "Iteration %d:\t %f\n", i, total_energy_[i]);
      
    }
    fclose(energy_pointer);
    
    const char* energy_vector_file = 
        fx_param_str(module_, "Evec", "energy_vector.csv");
    Matrix energy_vector_matrix;
    energy_vector_matrix.AliasColVector(energy_vector_);
    data::Save(energy_vector_file, energy_vector_matrix);
    
    fx_format_result(module_, "density_matrix_norm", "%g", 
                     density_matrix_frobenius_norm_);
                     
    fx_format_result(module_, "fock_max", "%g", fock_max_);
    fx_format_result(module_, "fock_min", "%g", fock_min_);
    fx_format_result(module_, "density_max", "%g", density_max_);
    fx_format_result(module_, "density_min", "%g", density_min_);
                     
    fx_format_result(module_, "nuclear_repulsion", "%g", 
                     nuclear_repulsion_energy_);
                     
    fx_format_result(module_, "one_electron_energy", "%g", 
                     one_electron_energy_);
    
    fx_format_result(module_, "two_electron_energy", "%g", 
                     two_electron_energy_);
    
    fx_format_result(module_, "num_iterations", "%d", current_iteration_);
    
    fx_format_result(module_, "total_energy", "%g", 
                     total_energy_[current_iteration_]);
    
    Matrix coulomb_out;
    Matrix exchange_out;
    
    coulomb_max_ = -DBL_INF;
    coulomb_min_ = DBL_INF;
    exchange_max_ = -DBL_INF;
    exchange_min_ = DBL_INF;
    
    integrals_.OutputFockMatrix(NULL, &coulomb_out, &exchange_out, NULL);
    
    for (index_t i = 0; i < number_of_basis_functions_; i++) {
    
      for (index_t j = i; j < number_of_basis_functions_; j++) {
      
        if (coulomb_out.ref(i,j) > coulomb_max_) {
          coulomb_max_ = coulomb_out.ref(i,j);
        }
        if (coulomb_out.ref(i,j) < coulomb_min_) {
          coulomb_min_ = coulomb_out.ref(i,j);
        }
        if (exchange_out.ref(i,j) > exchange_max_) {
          exchange_max_ = exchange_out.ref(i,j);
        }
        if (exchange_out.ref(i,j) < exchange_min_) {
          exchange_min_ = exchange_out.ref(i,j);
        }
        
      } // j
    
    } // i
    
    fx_format_result(module_, "coulomb_max", "%g", coulomb_max_);
    fx_format_result(module_, "coulomb_min", "%g", coulomb_min_);
    fx_format_result(module_, "exchange_max", "%g", exchange_max_);
    fx_format_result(module_, "exchange_min", "%g", exchange_min_);
    
  }
  
 public:
  
  /**
   * Compute the Hartree-Fock wavefunction for the given values of the 
   * integrals. 
   */
  void ComputeWavefunction() {
      
    fx_timer_start(module_, "SCF_Setup");
    Setup_();
    fx_timer_stop(module_, "SCF_Setup");
      
    fx_timer_start(module_, "SCF_Iterations");
    FindSCFSolution_();
    fx_timer_stop(module_, "SCF_Iterations");
    
    OutputResults_();
      
  } // ComputeWavefunction
  
  void PrintMatrices() {
    
    // These should be changed to print debug or something
    
    printf("Core Matrix:\n");
    core_matrix_.PrintDebug();
    
    printf("Coefficient matrix:\n");
    coefficient_matrix_.PrintDebug();
    
    printf("Change-of-basis matrix:\n");
    overlap_matrix_.PrintDebug();
    
    printf("Density matrix:\n");
    density_matrix_.PrintDebug();
    
    printf("Fock matrix:\n");
    fock_matrix_.PrintDebug();
    
    printf("Energy vector:\n");
    energy_vector_.PrintDebug();
    
    printf("Total energy:\n");
    ot::Print(total_energy_);  
      
  } // PrintMatrices
  
}; // class HFSolver


#endif // inclusion guards