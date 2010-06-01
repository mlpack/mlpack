/*
 *  shell_tree_impl.cc
 *  
 *
 *  Created by William March on 8/19/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#include "shell_tree_impl.h"


namespace shell_tree_impl {
  
  // this rearranges the list and returns the index of the midpoint
  index_t SelectListPartition(ArrayList<BasisShell*>& shells, double split_val, 
                              int split_dim, index_t begin, index_t count,
                              DHrectBound<2>* left_space, DRange* left_exp, 
                              DRange* left_mom, DHrectBound<2>* right_space, 
                              DRange* right_exp, DRange* right_mom, 
                              DRange* left_norms, DRange* right_norms,
                              ArrayList<index_t>* perm) {
    
    DEBUG_ASSERT(split_dim >= 0);
    DEBUG_ASSERT(split_dim <= 4);
    DEBUG_ASSERT(count >= 1);
    
    index_t left = begin;
    index_t right = begin + count - 1;
    
    for(;;) {
      
      double left_splitval;
      if (split_dim == 4) {
        //split on momentum
        left_splitval = shells[left]->total_momentum();
      }
      else if (split_dim == 3) {
        //split on exponent
        left_splitval = shells[left]->exp();
      }
      else {
        // split on space
        left_splitval = shells[left]->center()[split_dim];
      }
      
      while(left_splitval < split_val && left <= right) {
        
        // include the leftmost point in the left box
        
        *left_space |= shells[left]->center();
        *left_exp |= shells[left]->exp();
        *left_mom |= shells[left]->total_momentum();  
        
        for (index_t j = 0; j < shells[left]->num_functions(); j++) {
          *left_norms |= shells[left]->normalization_constant(j);
        }
        
        left++;
        
        if (split_dim == 4) {
          //split on momentum
          left_splitval = shells[left]->total_momentum();
        }
        else if (split_dim == 3) {
          //split on exponent
          left_splitval = shells[left]->exp();
        }
        else {
          // split on space
          left_splitval = shells[left]->center()[split_dim];
        }
        
        
      } // increment left
      
      
      double right_splitval;
      if (split_dim == 4) {
        //split on momentum
        right_splitval = shells[right]->total_momentum();
      }
      else if (split_dim == 3) {
        //split on exponent
        right_splitval = shells[right]->exp();
      }
      else {
        // split on space
        right_splitval = shells[right]->center()[split_dim];
      }
      
      while(right_splitval >= split_val && left <= right) {
        
        // include the rightmost point in the left box
        
        *right_space |= shells[right]->center();
        *right_exp |= shells[right]->exp();
        *right_mom |= shells[right]->total_momentum();    
        
        for (index_t j = 0; j < shells[right]->num_functions(); j++) {
          *right_norms |= shells[right]->normalization_constant(j);
        }
        
        right--;
        
        if (split_dim == 4) {
          //split on momentum
          right_splitval = shells[right]->total_momentum();
        }
        else if (split_dim == 3) {
          //split on exponent
          right_splitval = shells[right]->exp();
        }
        else {
          // split on space
          right_splitval = shells[right]->center()[split_dim];
        }
        
        
      } // decrement right
      
      if (left > right) {
        break;
      }
      
      // now the vectors need to be swapped
      
      eri::ArrayListSwapPointers(left, right, &shells);
      
      *left_space |= shells[left]->center();
      *left_exp |= shells[left]->exp();
      *left_mom |= shells[left]->total_momentum();
      
      for (index_t j = 0; j < shells[left]->num_functions(); j++) {
        *left_norms |= shells[left]->normalization_constant(j);
      }
      
      *right_space |= shells[right]->center();
      *right_exp |= shells[right]->exp();
      *right_mom |= shells[right]->total_momentum();
      
      for (index_t j = 0; j < shells[right]->num_functions(); j++) {
        *right_norms |= shells[right]->normalization_constant(j);
      }
      
      
      if (perm) {
        eri::ArrayListSwap(left, right, perm);
      }
      
      DEBUG_ASSERT(left <= right);
      right--;
      
    } // outer while
    
    DEBUG_ASSERT(left == right +1);
    DEBUG_ASSERT(left >= 0);
    
    
    return left;
    
  } //SelectListPartition
  

  void SelectSplit(ArrayList<BasisShell*>& shells, BasisShellTree* node, 
                   index_t leaf_size, ArrayList<index_t>* old_from_new) {
    
    BasisShellTree* left = NULL;
    BasisShellTree* right = NULL;
    
    double max_width = -1;
    index_t split_dim = BIG_BAD_NUMBER;
    double w;
    
    // are there different momenta?  
    if (!node->single_momentum()) {
     
      max_width = DBL_MAX;
      split_dim = 4;
      
    }
    else if (node->count() > leaf_size) {
     
      // for now, treating exponents equally to distances
      // in the future, replace this with some other function of the 
      // exponents
      //max_width = node->max_exponent() - node->min_exponent();
      max_width = node->exponents().width();
      split_dim = 3;
      
      for (index_t d = 0; d < 3; d++) {
        w = node->bound().get(d).width();
        if (w > max_width) {
          max_width = w;
          split_dim = d;
        }
      }
      
    } // not spliting on momenta, check leaf size
    
    
    if (max_width > 0.0) {
      
      double split_val;
      if (split_dim == 4) {
        split_val = node->momenta().mid();
      }
      else if (split_dim == 3) {
        split_val = node->exponents().mid();
      } else {
        split_val = node->bound().get(split_dim).mid();
      }
      
      
      left = new BasisShellTree();
      left->bound().Init(3);
      left->exponents().InitEmptySet();
      left->momenta().InitEmptySet();
      left->normalizations().InitEmptySet();
      
      right = new BasisShellTree();
      right->bound().Init(3);
      right->exponents().InitEmptySet();
      right->momenta().InitEmptySet();
      right->normalizations().InitEmptySet();
      
      // shells get rearranged here
      index_t split_col = SelectListPartition(shells, split_val, split_dim,
                                              node->begin(), node->count(),
                                              &(left->bound()), &(left->exponents()),
                                              &(left->momenta()), &(right->bound()),
                                              &(right->exponents()), &(right->momenta()),
                                              &(left->normalizations()), &(right->normalizations()),
                                              old_from_new);
      
      left->Init(node->begin(), split_col - node->begin());
      right->Init(split_col, node->begin() + node->count() - split_col);
      
      /*
      for (index_t i = 0; i < shells.size(); i++) {
        printf("momentum: %d\n", shells[i]->total_momentum());
      }
      */
      
      SelectSplit(shells, left, leaf_size, old_from_new);
      SelectSplit(shells, right, leaf_size, old_from_new);
      
    } // node is wide enough to split
    
    node->set_children(left, right);
        
  } // SelectSplit
  
  
  BasisShellTree* CreateShellTree(ArrayList<BasisShell*>& shells, 
                                  index_t leaf_size,
                                  ArrayList<index_t> *old_from_new,
                                  ArrayList<index_t> *new_from_old) {
    
    if (old_from_new) {
      old_from_new->Init(shells.size());
      
      for (index_t i = 0; i < shells.size(); i++) {
        (*old_from_new)[i] = i;
      }
      
    }
    
    BasisShellTree* node = new BasisShellTree();
    node->bound().Init(3);
    node->exponents().InitEmptySet();
    node->momenta().InitEmptySet();
    node->normalizations().InitEmptySet();
    
    // need to iniut these to right ranges
    for (index_t i = 0; i < shells.size(); i++) {
      
      node->bound() |= shells[i]->center();
      node->exponents() |= shells[i]->exp();
      node->momenta() |= shells[i]->total_momentum();
      
      for (index_t j = 0; j < shells[i]->num_functions(); j++) {
        node->normalizations() |= shells[i]->normalization_constant(j);
      }
      
    }
    
    node->Init(0, shells.size());
    
    SelectSplit(shells, node, leaf_size, old_from_new);
    
    
    if (new_from_old) {
     
      new_from_old->Init(shells.size());
      for (index_t i = 0; i < shells.size(); i++) {
        (*new_from_old)[(*old_from_new)[i]] = i;
      }
      
    }
    
    return node;
    
  } // CreateShellTree()


}