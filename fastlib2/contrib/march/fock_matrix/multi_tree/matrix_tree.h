/*
 *  matrix_tree.h
 *  
 *
 *  Created by William March on 8/24/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef MATRIX_TREE_H
#define MATRIX_TREE_H

#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "shell_tree_impl.h"


class MatrixTree {

private:
  
  /**
   * @brief The list of indices in the fock matrix this node is responsible for.
   */
  // how to set these correctly?  The shell tree doesn't know this, other than 
  // through the shells themselves. 
  // these are the concatenations of the matrix_indices_ arrays in the individual
  // shells that make up the underlying BasisShellTree nodes
  // I also need an invariant on whether to use the upper or lower triangle of 
  // the matrix, and which underlying BasisShellTree node corresponts to the rows
  // and which to the columns.
  ArrayList<index_t> row_indices_;
  ArrayList<index_t> col_indices_;
  
  /**
   * @brief Any approximations that need to be added to all matrix entries in 
   * this node.
   * This doesn't need to be passed down the tree, since the matrix reconstruction
   * code will take care of this.
   */
  double approx_val_;
  
  /**
   * @brief Upper and lower bounds on the Fock matrix entries (i.e. query bounds)
   */
  DRange fock_bounds_;
  
  /**
   * @brief Density matrix bounds (i.e. reference bounds)
   */
  DRange density_bounds_;
  
  /**
   * @brief The underlying BasisShellTree pointers
   */
  BasisShellTree* row_shells_;
  BasisShellTree* col_shells_;
  
  /**
   * @brief The children.
   */
  MatrixTree* left_;
  MatrixTree* right_;
  
  bool is_leaf_;
  
  Matrix* fock_entries_;
  
  bool on_diagonal_;
    
public:
  
  ArrayList<index_t>& row_indices() {
    return row_indices_;
  }
  
  ArrayList<index_t>& col_indices() {
    return col_indices_;
  }
  
  DRange& fock_bounds() {
    return fock_bounds_;
  } 
  
  DRange& density_bounds() {
    return density_bounds_;
  }
  
  double approx_val() const {
    return approx_val_;
  }
  
  void add_approx(double val) {
    approx_val_ += val;
  }
  
  void set_approx_val(double val) {
    approx_val_ = val;
  }
  
  MatrixTree* left() {
    return left_;
  }

  MatrixTree* right() {
    return right_;
  }
  
  BasisShellTree* row_shells() {
    return row_shells_;
  }

  BasisShellTree* col_shells() {
    return col_shells_;
  }
  
  bool is_leaf() const {
    return is_leaf_;
  }
  
  Matrix* fock_entries() {
    return fock_entries_; 
  }
  
  bool on_diagonal() const {
    return on_diagonal_;
  }
  
  void set_children(MatrixTree* left, MatrixTree* right) {
    
    left_ = left;
    right_ = right;
    //TODO: propagate approximations down here
    
  }
  
  // discovered that we can't split this node
  void make_leaf() {
    
    // it shouldn't already be a leaf
    DEBUG_ASSERT(!is_leaf_);
    
    is_leaf_ = true;
    
    fock_entries_ = new Matrix();
    fock_entries_->Init(row_indices_.size(), col_indices_.size());
    fock_entries_->SetZero(); 
    
  }
  
  void Init(BasisShellTree* rows, BasisShellTree* cols, 
            const ArrayList<BasisShell*>& shells, const Matrix& density) {
    
    row_shells_ = rows;
    col_shells_ = cols;
    
    // this is the invariant to preserve the upper triangular structure
    DEBUG_ASSERT(row_shells_->end() > col_shells_->begin());
    
    row_indices_.Init();
    for (index_t i = row_shells_->begin(); i < row_shells_->end(); i++) {
      row_indices_.AppendCopy(shells[i]->matrix_indices());
    }
    DEBUG_ASSERT(row_indices_.size() > 0);
    
    col_indices_.Init();
    for (index_t i = col_shells_->begin(); i < col_shells_->end(); i++) {
      col_indices_.AppendCopy(shells[i]->matrix_indices());
    }
    DEBUG_ASSERT(col_indices_.size() > 0);

    density_bounds_.InitEmptySet();
    for (index_t i = 0; i < row_indices_.size(); i++) {
      for (index_t j = 0; j < col_indices_.size(); j++) {
        density_bounds_ |= density.get(row_indices_[i], col_indices_[j]);
      }
    }
    
    on_diagonal_ = (rows == cols);
    
    // Alternately, could use the Scwartz upper bound times the total number
    // of references
    // If there is a negative density matrix lower bound, then the lower bound
    // will be the density lower bound times the Schwartz bound.
    // How am I going to update this as I go?  I think I'll need a global upper 
    // and lower bound integral, then the bound at any time is the current approximations
    // plus the current computation plus the global bound times the number of 
    // pending references.  
    fock_bounds_.InitUniversalSet();
    
    approx_val_ = 0.0;
    
    // These start as NULL since we don't perform the split when creating the
    // tree node
    left_ = NULL;
    right_ = NULL;
    // This determines whether the node could be further split
    is_leaf_ = (row_shells_->is_leaf() && col_shells_->is_leaf());
    
    if (is_leaf_) {
      fock_entries_ = new Matrix();
      fock_entries_->Init(row_indices_.size(), col_indices_.size());
      fock_entries_->SetZero();
    }
    else {
      fock_entries_ = NULL; 
    }
    
  } // Init()
  
  void Print() {
    
    //row_indices_.Print("row_indices");
    //col_indices_.Print("col_indices");
    printf("row begin: %d, row count: %d, row end: %d\n", row_shells_->begin(),
           row_shells_->count(), row_shells_->end());
    printf("col begin: %d, col count: %d, col end: %d\n", col_shells_->begin(),
           col_shells_->count(), col_shells_->end());
    printf("density bounds: (%g, %g)\n", density_bounds_.lo, density_bounds_.hi);
    printf("left child: %p, right child: %p\n", left_, right_);
    printf("\n");
    
  }
    
  
}; // class MatrixTree

#endif 

