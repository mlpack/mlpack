/**
 * Computes statistics on difference between two input fock matrices.  
 */

#ifndef FOCK_MATRIX_COMPARISON_H
#define FOCK_MATRIX_COMPARISON_H


#include "fastlib/fastlib.h"

class FockMatrixComparison {

 private:
  
  // Used for computing the average difference between the matrices
  double total_diff_;
  
  // The maximum difference between the matrices
  double max_diff_;
  
  // The row and column index of the maximum difference
  index_t max_index_row_;
  index_t max_index_col_;
  
  Matrix mat1_;
  Matrix mat2_;
  
  fx_module* my_mod_;
  
  fx_module* mod1_;
  fx_module* mod2_;
  
  index_t num_entries_;

 public:

  void Init(fx_module* mod1, const Matrix& mat1, fx_module* mod2, 
            const Matrix& mat2, fx_module* my_mod);

  void Compare();
  
  void Destruct();

}; // class FockMatrixComparison