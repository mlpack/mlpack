/*
 *  shell_tree_test.cc
 *  
 *
 *  Created by William March on 8/20/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */


#include "contrib/march/fock_matrix/fock_impl/eri.h"
#include "shell_tree_impl.h"
#include "basis_shell_tree.h"

int main (int argc, char* argv[]) {
  
  fx_module* root_mod = fx_init(argc, argv, NULL);
  eri::ERIInit();
  
  Matrix centers;
  const char* centers_file = fx_param_str_req(root_mod, "centers");
  data::Load(centers_file, &centers);
  
  Matrix exp_mat;
  const char* exp_file = fx_param_str_req(root_mod, "exponents");
  data::Load(exp_file, &exp_mat);
  Vector exponents;
  exponents.Copy(exp_mat.ptr(), centers.n_cols());
  
  if (centers.n_cols() != exp_mat.n_cols()) {
    FATAL("Number of basis centers must equal number of exponents.\n");
  }
  
  Matrix momenta;
  if (fx_param_exists(root_mod, "momenta")) {
    const char* momenta_file = fx_param_str_req(root_mod, "momenta");
    data::Load(momenta_file, &momenta);
  }
  else {
    momenta.Init(1, centers.n_cols());
    momenta.SetAll(0);
  }
  Vector mom_vec;
  mom_vec.Copy(momenta.ptr(), centers.n_cols());
  
  ArrayList<BasisShell> shells;
  index_t num_functions = eri::CreateShells(centers, exponents, mom_vec, &shells);
  
  ArrayList<BasisShell*> shell_ptrs;
  shell_ptrs.Init(shells.size());
  for (int i = 0; i < shells.size(); i++) {
    shell_ptrs[i] = &(shells[i]);
  }
  
  ArrayList<index_t> old_from_new;
  BasisShellTree* tree = shell_tree_impl::CreateShellTree(shell_ptrs, 1, 
                                                          &old_from_new,
                                                          NULL);
  
  tree->Print();
  old_from_new.Print("old from new");
  
  return 0;
  
} 
