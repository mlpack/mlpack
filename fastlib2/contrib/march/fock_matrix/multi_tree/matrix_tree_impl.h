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
  
  void FormDenseMatrix(MatrixTree* root, Matrix* fock_out);
  
  MatrixTree* CreateMatrixTree();
  
  void SplitMatrixTree(MatrixTree* node);
  
}; // namespace matrix_tree_impl


#endif