/*
 *  shell_tree_impl.h
 *  
 *
 *  Created by William March on 8/19/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef SHELL_TREE_IMPL_H
#define SHELL_TREE_IMPL_H

#include "fastlib/fastlib.h"
#include "basis_shell_tree.h"

namespace shell_tree_impl {
  
  /**
   * Once the split dimension and value have been chosen, this sorts the shell 
   * list accordingly
   *
   * In kdtree_impl, this sets up the child bounds as well.  
   */
  index_t SelectListPartition(ArrayList<BasisShell*>& shells, double split_val, 
                              int split_dim, index_t begin, index_t count,
                              DHrectBound<2>* left_space, DRange* left_exp, 
                              DRange* left_mom, DHrectBound<2>* right_space, 
                              DRange* right_exp, DRange* right_mom, 
                              DRange* left_norms, DRange* right_norms,
                              ArrayList<index_t>* perm);
  
  /**
   * Picks which dimension to split and does it
   */
  void SelectSplit(ArrayList<BasisShell*>& shells, BasisShellTree* node, 
                   index_t leaf_size);
  
  /**
   * Use this to create the shell tree
   */
  BasisShellTree* CreateShellTree(ArrayList<BasisShell*>& shells, index_t leaf_size,
                                  ArrayList<index_t> *old_from_new,
                                  ArrayList<index_t> *new_from_old);
  
  
  
} // namespace shell_tree_impl


#endif