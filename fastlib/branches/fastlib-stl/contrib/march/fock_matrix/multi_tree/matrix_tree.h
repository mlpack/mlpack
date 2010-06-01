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
   * These are the global indices - i.e. the ones to reference the density with
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
   * This does need to be passed down the tree in order to prune accurately.
   */
  /*
  double coulomb_approx_val_;
  double exchange_approx_val_;
  */
  
  double approx_val_;
  
  /**
   * @brief Upper and lower bounds on the query
   */
  DRange entry_bounds_;
  
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
  
  Matrix* entries_;
  
  bool on_diagonal_;
  
  double remaining_epsilon_;
  
  // The number of reference pairs this query hasn't yet accounted for
  int remaining_references_;
  
  // the number of pairs of SHELLS contained in this node
  // this counts those below the diagonal as well
  int num_pairs_;
    
public:
  
  ArrayList<index_t>& row_indices() {
    return row_indices_;
  }
  
  ArrayList<index_t>& col_indices() {
    return col_indices_;
  }
  
  DRange& entry_bounds() {
    return entry_bounds_;
  } 
  
  DRange& density_bounds() {
    return density_bounds_;
  }
  
  /*
  double coulomb_approx_val() const {
    return coulomb_approx_val_;
  }

  double exchange_approx_val() const {
    return exchange_approx_val_;
  }
  
  void add_coulomb_approx(double val) {
    coulomb_approx_val_ += val;
  }
  
  void add_exchange_approx(double val) {
    exchange_approx_val_ += val;
  }
  
  void set_coulomb_approx_val(double val) {
    coulomb_approx_val_ = val;
  }

  void set_exchange_approx_val(double val) {
    exchange_approx_val_ = val;
  }
   */
  
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

  void set_rows(BasisShellTree* r) {
    row_shells_ = r;
  }
  
  BasisShellTree* col_shells() {
    return col_shells_;
  }
  
  void set_cols(BasisShellTree* c) {
    col_shells_ = c;
  }
  
  bool is_leaf() const {
    return is_leaf_;
  }
  
  /*
  Matrix* coulomb_entries() {
    return coulomb_entries_; 
  }

  Matrix* exchange_entries() {
    return exchange_entries_; 
  }
  */
  
  Matrix* entries() {
    return entries_;
  }
  
  bool on_diagonal() const {
    return on_diagonal_;
  }
  
  double remaining_epsilon() const {
    return remaining_epsilon_;
  }
  
  void set_remaining_epsilon(double eps) {
    DEBUG_ASSERT(eps >= 0.0);
    remaining_epsilon_ = eps;
  }

  int remaining_references() const {
    return remaining_references_;
  }
  
  void set_remaining_references(int ref) {
    DEBUG_ASSERT(ref >= 0);
    remaining_references_ = ref;
  }
  
  int num_pairs() const {
    return num_pairs_;
  }
  
  void set_num_pairs(int pairs) {
    DEBUG_ASSERT(pairs >= 0);
    num_pairs_ = pairs;
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
    // TODO: should I update the bounds here
    // might not matter, because it's a leaf I don't need to care - only 
    // doing the base case
    // I do need to update the row and column indices
    
    is_leaf_ = true;
    
    // this should work here, since I should only be making "square" nodes
    // into leaves
    on_diagonal_ = (row_shells_ == col_shells_);
    
    /*
    num_pairs_ = row_indices_.size() * col_indices_.size();
    if (!on_diagonal_) {
      // account for symmetry
      num_pairs_ *= 2;
    }
    */
    
    /*
    coulomb_entries_ = new Matrix();
    coulomb_entries_->Init(row_indices_.size(), col_indices_.size());
    coulomb_entries_->SetZero();
    exchange_entries_ = new Matrix();
    exchange_entries_->Init(row_indices_.size(), col_indices_.size());
    exchange_entries_->SetZero();
    */
    entries_ = new Matrix();
    entries_->Init(row_indices_.size(), col_indices_.size());
    entries_->SetZero();
    
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
    
    // this doesn't account for the possibilty of a rectangular node 
    // that straddles the diagonal
    // I can determine this from the begin and end of the BasisShellTrees
    on_diagonal_ = ((row_shells_->begin() >= col_shells_->begin()) 
                    && (row_shells_->begin() < col_shells_->end())) ||
                   ((col_shells_->begin() >= row_shells_->begin()) 
                    && (col_shells_->begin() < row_shells_->end()));
    
    // Alternately, could use the Scwartz upper bound times the total number
    // of references
    // If there is a negative density matrix lower bound, then the lower bound
    // will be the density lower bound times the Schwartz bound.
    // How am I going to update this as I go?  I think I'll need a global upper 
    // and lower bound integral, then the bound at any time is the current approximations
    // plus the current computation plus the global bound times the number of 
    // pending references.  
    entry_bounds_.InitUniversalSet();
    
    /*
    coulomb_approx_val_ = 0.0;
    exchange_approx_val_ = 0.0;
    */
    approx_val_ = 0.0;
    
    // These start as NULL since we don't perform the split when creating the
    // tree node
    left_ = NULL;
    right_ = NULL;
    // This determines whether the node could be further split
    is_leaf_ = (row_shells_->is_leaf() && col_shells_->is_leaf());
    
    if (is_leaf_) {
      /*
      coulomb_entries_ = new Matrix();
      coulomb_entries_->Init(row_indices_.size(), col_indices_.size());
      coulomb_entries_->SetZero();
      exchange_entries_ = new Matrix();
      exchange_entries_->Init(row_indices_.size(), col_indices_.size());
      exchange_entries_->SetZero();
      */
      entries_ = new Matrix();
      entries_->Init(row_indices_.size(), col_indices_.size());
      entries_->SetZero();
      
    }
    else {
      /*
      coulomb_entries_ = NULL; 
      exchange_entries_ = NULL;
       */
      entries_ = NULL;
      
    }
    
    
    DRange row_range;
    row_range.Init((double)(row_shells_->begin()), (double)(row_shells_->end()));
    DRange col_range;
    col_range.Init((double)(col_shells_->begin()), (double)(col_shells_->end()));
    row_range &= col_range;
    //int num_on_diagonal = (int)(row_range.width()) * (int)(row_range.width());
    //int num_off_diagonal = row_shells_->count() * col_shells_->count() - num_on_diagonal;
    
    if (!on_diagonal_) {
      // off diagonal, count the entire thing twice
      //num_pairs_ = row_indices_.size() * col_indices_.size() * 2;
      //printf("Off diagonal pairs: %d\n", num_pairs_);
      num_pairs_ = row_shells_->count() * col_shells_->count() * 2;
    }
    else {
      // this is a rectangle on the diagonal
      // this node should never be involved in a base case. 
      DRange row_range;
      row_range.Init((double)(row_shells_->begin()), (double)(row_shells_->end()));
      DRange col_range;
      col_range.Init((double)(col_shells_->begin()), (double)(col_shells_->end()));
      row_range &= col_range;
      // This should be true since the node is on the diagonal
      DEBUG_ASSERT(row_range.width() > 0.0);
      int num_on_diagonal = (int)(row_range.width()) * (int)(row_range.width());
      int num_off_diagonal = row_shells_->count() * col_shells_->count() - num_on_diagonal;
      
      //printf("On diagonal pairs: On: %d, Off: %d\n", num_on_diagonal, num_off_diagonal);
      num_pairs_ = 2 * num_off_diagonal + num_on_diagonal;
    }
     
    
  } // Init()
  
  ~MatrixTree() {
  
    if (left_) {
      delete left_;
      delete right_;
    }
    
    if (entries_) {
      delete entries_;
    }
    
  }
    
  void Print() {
    
    //row_indices_.Print("row_indices");
    //col_indices_.Print("col_indices");
    printf("row begin: %d, row count: %d, row end: %d\n", row_shells_->begin(),
           row_shells_->count(), row_shells_->end());
    printf("col begin: %d, col count: %d, col end: %d\n", col_shells_->begin(),
           col_shells_->count(), col_shells_->end());
    printf("on diagonal: %d\n", on_diagonal());
    printf("density bounds: (%g, %g)\n", density_bounds_.lo, density_bounds_.hi);
    printf("left child: %p, right child: %p\n", left_, right_);
    printf("\n");
    
    if (left_) {
      left_->Print();
      right_->Print();
    }
    
  }
    
  
}; // class MatrixTree

#endif 

