/*
 *  matrix_tree_impl.h
 *  
 *
 *  Created by William March on 8/24/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MATRIX_TREE_IMPL_H
#define MATRIX_TREE_IMPL_H

#include "fastlib/fastlib.h"
#include "matrix_tree.h"

namespace matrix_tree_impl {
  
  void FormDenseMatrixHelper(MatrixTree* node, Matrix* mat_out, double approx);
  
  void FormDenseMatrix(MatrixTree* root, Matrix* mat_out);
  
  MatrixTree* CreateMatrixTree(BasisShellTree* shell_root, 
                               const ArrayList<BasisShell*>& shells,
                               const Matrix& density);
  
  success_t SplitMatrixTree(MatrixTree* node, const ArrayList<BasisShell*>& shells,
                       const Matrix& density);
  
}; // namespace matrix_tree_impl


#endif