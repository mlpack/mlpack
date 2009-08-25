/*
 *  matrix_tree_impl.cc
 *  
 *
 *  Created by William March on 8/24/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "matrix_tree_impl.h"

namespace matrix_tree_impl {

  void FormDenseMatrixHelper(MatrixTree* node, Matrix* mat_out, 
                             double approx_val) {
    
    double this_approx = node->approx_val() + approx_val;
    
    if (node->left()) {
      // recursively build the matrix
      // how to pass the submatrices around? 
      FormDenseMatrixHelper(node->left(), mat_out, this_approx);
      
      FormDenseMatrixHelper(node->right(), mat_out, this_approx);
      
    } // not leaf
    else if (node->fock_entries()){
      
      // Iterate over the fock_entries matrix
      for (index_t i = 0; i < node->row_indices().size(); i++) {
        
        index_t row_ind = node->row_indices()[i];
        
        for (index_t j = 0; j < node->col_indices().size(); j++) {
          
          index_t col_ind = node->col_indices()[j];
          double this_val = mat_out->get(row_ind, col_ind) + this_approx 
          + node->fock_entries()->get(i,j);
          
          mat_out->set(row_ind, col_ind, this_val);
          
          // handle below the diagonal
          if (!node->on_diagonal()) {
            mat_out->set(col_ind, row_ind, this_val); 
          }
          
        } // for j
      } // for i
      
    } // leaf (with matrix)
    else {
      // no base case matrix
      
      // Iterate over the fock_entries matrix
      for (index_t i = 0; i < node->row_indices().size(); i++) {
        
        index_t row_ind = node->row_indices()[i];
        
        for (index_t j = 0; j < node->col_indices().size(); j++) {
          
          index_t col_ind = node->col_indices()[j];
          double this_val = mat_out->get(row_ind, col_ind) + this_approx;
          
          mat_out->set(row_ind, col_ind, this_val);
          
          // handle below the diagonal
          if (!node->on_diagonal()) {
            // matrix should always be symmetric
            DEBUG_ASSERT(mat_out->get(row_ind, col_ind) 
                         == mat_out->get(col_ind, row_ind));
            mat_out->set(col_ind, row_ind, this_val); 
          }
          
        } // for j
      } // for i
      
    } // was pruned, so no base case to iterate over
    
  } // FormDensityMatrixHelper()
  
  void FormDenseMatrix(MatrixTree* root, Matrix* fock_out) {
    
    // The root should represent a square matrix
    DEBUG_ASSERT(root->row_indices().size() == root->col_indices().size());
    
    fock_out->Init(root->row_indices().size(), root->row_indices().size());
    fock_out->SetZero();
    
    FormDenseMatrixHelper(root, fock_out, 0.0);
    
  } // FormDenseMatrix()
  
  MatrixTree* CreateMatrixTree(BasisShellTree* shell_root, 
                               const ArrayList<BasisShell*>& shells,
                               const Matrix& density) {
    
    MatrixTree* root = new MatrixTree();
    root->Init(shell_root, shell_root, shells, density);
    
    return root;
    
  } // CreateMatrixTree()
  
  void SplitMatrixTree(MatrixTree* node, const ArrayList<BasisShell*>& shells,
                       const Matrix& density) {
    
    DEBUG_ASSERT(!node->is_leaf());
    
    // which BasisShellTree node to split?  
    // just make sure that row_shells_.end() > col_shells_.begin()
    
    // pick whether to split rows or columns
    // previous code used the one higher up in the tree and broke ties for rows
    // for simplicity
    BasisShellTree* rows = node->row_shells();
    BasisShellTree* cols = node->col_shells();
    
    DEBUG_ASSERT(cols->height() > 0 || rows->height() > 0);
    
    if (rows->height() >= cols->height()) {
      // split rows
      
      if (rows->left()->end() > cols->begin()) {
        // then we can split the rows like normal 
      }
      else if (rows->right()->height() == 0) {
        // the rows are too low to split
        // this is either a leaf, or we try to split the columns
        // I don't think splitting the columns would work
        
        node->make_leaf();
        return;
        
      }
      else {
        // split rows->right()
        rows = rows->right();
      }
      
      MatrixTree* left_child = new MatrixTree();
      MatrixTree* right_child = new MatrixTree();
      
      left_child->Init(rows->left(), cols, shells, density);
      right_child->Init(rows->right(), cols, shells, density);
      
      node->set_children(left_child, right_child);
      
    } // splitting row shells
    else {
      // splitting the cols
      
      if (rows->end() > cols->right()->begin()) {
        // fine to split cols 
      }
      else if (cols->left()->height() == 0) {
        // cols are too low to split twice, needs to be a leaf
        
        node->make_leaf();
        return;
        
      }
      else {
        // split cols left
        cols = cols->left();
      }
      
      MatrixTree* left_child = new MatrixTree();
      MatrixTree* right_child = new MatrixTree();
      
      left_child->Init(rows, cols->left(), shells, density);
      left_child->add_approx(node->approx_val());
      right_child->Init(rows, cols->right(), shells, density);
      right_child->add_approx(node->approx_val());
      // it's been passed down to the children
      node->set_approx_val(0.0);
      
      node->set_children(left_child, right_child);
      
      
    } // splitting col shells
    
  } // SplitMatrixTree()
  
} // namespace matrix_tree_impl


