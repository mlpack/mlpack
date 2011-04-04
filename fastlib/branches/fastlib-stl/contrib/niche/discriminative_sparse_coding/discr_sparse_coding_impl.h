#ifndef INSIDE_DISCR_SPARSE_CODING_H
#error "This is not a public header file!"
#endif

void DiscrSparseCoding::Init(const mat& X, const vec& y, u32 n_atoms,
			     double lambda_1, double lambda_2) {
  X_ = X;
  y_ = y;
  
  n_dims_ = X.n_rows;
  n_points_ = X.n_cols;
  
  n_atoms_ = n_atoms;
  D_ = mat(n_dims_, n_atoms_);
  
  lambda_1_ = lambda_1;
  lambda_2_ = lambda_2;
}




#include "discr_sparse_coding.h"
