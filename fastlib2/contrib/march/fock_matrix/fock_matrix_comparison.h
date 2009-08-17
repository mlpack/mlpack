/**
 * Computes statistics on difference between two input fock matrices.  
 */

#ifndef FOCK_MATRIX_COMPARISON_H
#define FOCK_MATRIX_COMPARISON_H

#include "fastlib/fastlib.h"
#include "fock_impl/eri.h"
#include "fock_impl/basis_shell.h"

class FockMatrixComparison {

 private:
  
  // Used for computing the average difference between the matrices
  double total_diff_F_;
  double total_diff_J_;
  double total_diff_K_;
  
  // The maximum difference between the matrices
  double max_diff_F_;
  double max_diff_J_;
  double max_diff_K_;
  
  double max_rel_F_;
  double max_rel_J_;
  double max_rel_K_;
  
  // root mean square error
  double rms_F_;
  double rms_J_;
  double rms_K_;
  
  // The row and column index of the maximum difference
  index_t max_index_row_F_;
  index_t max_index_col_F_;

  index_t max_index_row_J_;
  index_t max_index_col_J_;

  index_t max_index_row_K_;
  index_t max_index_col_K_;
  
  index_t max_abs_index_row_J_;
  index_t max_abs_index_col_J_;

  
  bool compare_fock_;
  bool compare_coulomb_;
  bool compare_exchange_;
  bool compare_energies_;
  
  Matrix* G_mat_;
  Matrix* J_mat_;
  Matrix* K_mat_;
  
  Matrix fock_mat_;
  
  Matrix* naive_G_mat_;
  Matrix* naive_J_mat_;
  Matrix* naive_K_mat_;
  
  Matrix naive_fock_mat_;
  
  Vector naive_energies_;
  Vector comp_energies_;
  
  double naive_total_energy_;
  double comp_total_energy_;
  
  Matrix core_hamiltonian_;
  Matrix overlap_matrix_;
  Matrix change_of_basis_matrix_;
  
  Matrix centers_;
  Matrix nuclear_centers_;
  Vector nuclear_charges_;
  Matrix density_;
  Vector momenta_;
  Vector exponents_;
  
  fx_module* my_mod_;
  
  fx_module* approx_mod_;
  fx_module* naive_mod_;
  
  index_t num_entries_;
  
  std::string core_string_;
  std::string change_string_;
  
  double rel_error_cutoff_;
  
  void ComputeCoreMatrices_();
  
  void ComputeChangeOfBasisMatrix_();
  
  void DiagonalizeFockMatrix_();
  
  void CompareEigenvalues_();

 public:

  void Init(fx_module* mod1, Matrix** mat1, fx_module* mod2, 
            Matrix** mat2, const Matrix& centers, const Matrix& exp, 
            const Matrix& momenta, const Matrix& density, 
            Matrix* nuclear_centers, const Matrix& nuclear_charges, 
            fx_module* my_mod, const char* one_electron_name, 
            const char* change_of_basis_name);

  void Compare();
  
  void Destruct();

}; // class FockMatrixComparison

#endif