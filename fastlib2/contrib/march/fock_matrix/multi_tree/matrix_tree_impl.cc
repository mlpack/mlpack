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

  void FormDenseMatrixHelper(MatrixTree* node, Matrix* coul_out, Matrix* exc_out,
                             double coul_approx, double exc_approx) {
    
    double this_coul_approx = node->coulomb_approx_val() + coul_approx;
    double this_exc_approx = node->exchange_approx_val() + exc_approx;
    
    //printf("coul_approx: %g, exc_approx: %g\n", this_coul_approx, this_exc_approx);
    
    if (node->left()) {
      // recursively build the matrix
      // how to pass the submatrices around? 
      FormDenseMatrixHelper(node->left(), coul_out, exc_out, this_coul_approx,
                            this_exc_approx);
      FormDenseMatrixHelper(node->right(), coul_out, exc_out, this_coul_approx,
                            this_exc_approx);
      
    } // not leaf
    else if (node->coulomb_entries()){
      
      //node->coulomb_entries()->PrintDebug("Coulomb Base");
      
      
      DEBUG_ASSERT(node->exchange_entries());
      
      // Iterate over the fock_entries matrix
      for (index_t i = 0; i < node->row_indices().size(); i++) {
        
        index_t row_ind = node->row_indices()[i];
        
        for (index_t j = 0; j < node->col_indices().size(); j++) {
          
          index_t col_ind = node->col_indices()[j];
          double coul_val = coul_out->get(row_ind, col_ind) + this_coul_approx 
                            + node->coulomb_entries()->get(i,j);
          double exc_val = exc_out->get(row_ind, col_ind) + this_exc_approx 
                           + node->exchange_entries()->get(i,j);
          
          //printf("i: %d, j: %d, row_end: %d, col_ind: %d, prev_val: %g, new_val: %g\n", i, j, row_ind, col_ind, 
          //       coul_out->get(row_ind, col_ind), node->coulomb_entries()->get(i,j));
          coul_out->set(row_ind, col_ind, coul_val);
          exc_out->set(row_ind, col_ind, exc_val);
          
          // handle below the diagonal
          if (!(node->on_diagonal())) {
            coul_out->set(col_ind, row_ind, coul_val); 
            exc_out->set(col_ind, row_ind, exc_val); 
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
          double coul_val = coul_out->get(row_ind, col_ind) + this_coul_approx;
          double exc_val = exc_out->get(row_ind, col_ind) + this_exc_approx;
          
          coul_out->set(row_ind, col_ind, coul_val);
          exc_out->set(row_ind, col_ind, exc_val);
          
          // handle below the diagonal
          if (!node->on_diagonal()) {
            coul_out->set(col_ind, row_ind, coul_val); 
            exc_out->set(col_ind, row_ind, exc_val);
          }
          
        } // for j
      } // for i
      
    } // was pruned, so no base case to iterate over
    
  } // FormDensityMatrixHelper()
  
  void FormDenseMatrix(MatrixTree* root, Matrix* coulomb_out, 
                       Matrix* exchange_out) {
    
    // The root should represent a square matrix
    DEBUG_ASSERT(root->row_indices().size() == root->col_indices().size());
    
    coulomb_out->Init(root->row_indices().size(), root->row_indices().size());
    coulomb_out->SetZero();
    exchange_out->Init(root->row_indices().size(), root->row_indices().size());
    exchange_out->SetZero();
    
    //root->coulomb_entries()->PrintDebug("Coulomb Entries");
    //root->exchange_entries()->PrintDebug("Exchange Entries");
    
    FormDenseMatrixHelper(root, coulomb_out, exchange_out, 0.0, 0.0);
    
  } // FormDenseMatrix()
  
  MatrixTree* CreateMatrixTree(BasisShellTree* shell_root, 
                               const ArrayList<BasisShell*>& shells,
                               const Matrix& density) {
    
    MatrixTree* root = new MatrixTree();
    root->Init(shell_root, shell_root, shells, density);
    
    return root;
    
  } // CreateMatrixTree()
  
  success_t SplitMatrixTree(MatrixTree* node, const ArrayList<BasisShell*>& shells,
                            const Matrix& density) {
    
    DEBUG_ASSERT(!node->is_leaf());
    
    // which BasisShellTree node to split?  
    // just make sure that row_shells_.end() > col_shells_.begin()
    
    // pick whether to split rows or columns
    // previous code used the one higher up in the tree and broke ties for rows
    // for simplicity
    BasisShellTree* rows = node->row_shells();
    BasisShellTree* cols = node->col_shells();
    
    MatrixTree* left_child;
    MatrixTree* right_child;
    
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
        //printf("Making leaf\n");
        node->set_rows(rows->right());
        node->row_indices().Clear();
        for (index_t i = node->row_shells()->begin(); 
             i < node->row_shells()->end(); i++) {
            node->row_indices().AppendCopy(shells[i]->matrix_indices());
        }
        node->make_leaf();
        return SUCCESS_FAIL;
        
      }
      else {
        // split rows->right()
        rows = rows->right();
      }
      
      left_child = new MatrixTree();
      right_child = new MatrixTree();
      
      if (cols->left()->end() < cols->end()) {
        //printf("Double row split\n");
        left_child->Init(rows->left(), cols->left(), shells, density);
      }
      else {
        left_child->Init(rows->left(), cols, shells, density);
      }
      right_child->Init(rows->right(), cols, shells, density);
      
    } // splitting row shells
    else {
      // splitting the cols
      
      if (rows->end() > cols->right()->begin()) {
        // fine to split cols 
        
      }
      else if (cols->left()->height() == 0) {
        // cols are too low to split twice, needs to be a leaf
        
        //printf("Making leaf\n");
        node->set_cols(cols->left());
        node->col_indices().Clear();
        for (index_t i = node->col_shells()->begin(); 
             i < node->col_shells()->end(); i++) {
            node->col_indices().AppendCopy(shells[i]->matrix_indices());
        }
        node->make_leaf();
        return SUCCESS_FAIL;
        
      }
      else {
        // split cols left
        cols = cols->left();
      }
      
      
      
      left_child = new MatrixTree();
      right_child = new MatrixTree();
      
      left_child->Init(rows, cols->left(), shells, density);
      // have some nodes above the diagonal
      // split the cols again?
      // NO, in place of cols->right() below, I want cols->right()->right()?
      // what about if right is too shallow for this?
      if (cols->right()->begin() > rows->begin()) {
        //printf("Double col split\n");
        right_child->Init(rows->right(), cols->right(), shells, density);
      }
      else {
        right_child->Init(rows, cols->right(), shells, density);
      }
      
    } // splitting col shells
    
    // Taking the query passing out for now
    /*
    left_child->set_remaining_epsilon(node->remaining_epsilon());
    left_child->set_remaining_references(node->remaining_references());
    left_child->add_coulomb_approx(node->coulomb_approx_val());
    left_child->add_exchange_approx(node->exchange_approx_val());
    
    right_child->set_remaining_epsilon(node->remaining_epsilon());
    right_child->set_remaining_references(node->remaining_references());
    right_child->add_coulomb_approx(node->coulomb_approx_val());
    right_child->add_exchange_approx(node->exchange_approx_val());
    
    
    node->set_coulomb_approx_val(0.0);
    node->set_exchange_approx_val(0.0);
    */
    
    /*
    int num_left_pairs = left_child->row_shells()->count() 
                         * left_child->col_shells()->count();
    int num_right_pairs = right_child->row_shells()->count() 
                          * right_child->col_shells()->count();
    if (!(node->on_diagonal())) {
      num_left_pairs *= 2;
      num_right_pairs *= 2;
    }
    else if (left_child->on_diagonal() && right_child->on_diagonal()){
      // counts are already correct
    }
    else if (left_child->on_diagonal()){
      num_right_pairs *= 2;
    }
    else {
      DEBUG_ASSERT(right_child->on_diagonal());
      num_left_pairs *= 2;
    }
    //
    left_child->set_num_pairs(num_left_pairs);
    right_child->set_num_pairs(num_right_pairs);
     */
    DEBUG_ASSERT(node->num_pairs() == left_child->num_pairs() + right_child->num_pairs());
    node->set_children(left_child, right_child);
    
    return SUCCESS_PASS;
    
  } // SplitMatrixTree()
  
} // namespace matrix_tree_impl


