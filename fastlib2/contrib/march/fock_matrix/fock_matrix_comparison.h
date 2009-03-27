/**
 * Computes statistics on difference between two input fock matrices.  
 */

#ifndef FOCK_MATRIX_COMPARISON_H
#define FOCK_MATRIX_COMPARISON_H


#include "fastlib/fastlib.h"

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
  
  // The row and column index of the maximum difference
  index_t max_index_row_F_;
  index_t max_index_col_F_;

  index_t max_index_row_J_;
  index_t max_index_col_J_;

  index_t max_index_row_K_;
  index_t max_index_col_K_;

  
  bool compare_fock_;
  bool compare_coulomb_;
  bool compare_exchange_;
  
  Matrix* F_mat_;
  Matrix* J_mat_;
  Matrix* K_mat_;
  
  Matrix* naive_F_mat_;
  Matrix* naive_J_mat_;
  Matrix* naive_K_mat_;
  
  fx_module* my_mod_;
  
  fx_module* approx_mod_;
  fx_module* naive_mod_;
  
  index_t num_entries_;

 public:

  void Init(fx_module* mod1, Matrix** mat1, fx_module* mod2, 
            Matrix** mat2, fx_module* my_mod);

  void Compare();
  
  void Destruct();

}; // class FockMatrixComparison

#endif