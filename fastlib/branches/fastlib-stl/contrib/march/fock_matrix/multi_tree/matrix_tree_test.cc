/*
 *  matrix_tree_test.cc
 *  
 *
 *  Created by William March on 8/25/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */


#include "matrix_tree_impl.h"
#include "fastlib/fastlib.h"
#include "contrib/march/fock_matrix/chem_reader/chem_reader.h"


int main(int argc, char* argv[]) {

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
  
  std::string density_str;
  Matrix density;
  
  ArrayList<BasisShell> shells;
  index_t num_functions = eri::CreateShells(centers, exponents, mom_vec, &shells);
  
  if (fx_param_exists(root_mod, "density")) {
    
    const char* density_file = fx_param_str_req(root_mod, "density");
    density_str = density_file;
    
    size_t density_ext = density_str.find_last_of(".");
    if (!strcmp("qcmat", 
                density_str.substr(density_ext+1, std::string::npos).c_str())) {
      
      printf("Reading QChem Style Density Matrix.\n");
      chem_reader::ReadQChemDensity(density_file, &density, num_functions);
      // QC density matrices are only for alpha electrons
      la::Scale(2.0, &density);
      
    }
    else {
      
      printf("Reading csv Density Matrix.\n");
      data::Load(density_file, &density);
      
    }
  }
  else {
    density.Init(num_functions, num_functions);
    density.SetAll(1.0);
    density_str = "default";
  }
  
  
  ArrayList<BasisShell*> shell_ptrs;
  shell_ptrs.Init(shells.size());
  for (int i = 0; i < shells.size(); i++) {
    shell_ptrs[i] = &(shells[i]);
  }
  
  ArrayList<index_t> old_from_new;
  BasisShellTree* tree = shell_tree_impl::CreateShellTree(shell_ptrs, 1, 
                                                          &old_from_new,
                                                          NULL);
  
  MatrixTree* matrix_tree = matrix_tree_impl::CreateMatrixTree(tree, shell_ptrs,
                                                              density);
  
  
  matrix_tree->Print();
  
  matrix_tree_impl::SplitMatrixTree(matrix_tree, shell_ptrs, density);
  
  matrix_tree->Print();
  matrix_tree->left()->Print();
  matrix_tree->right()->Print();
  
  matrix_tree_impl::SplitMatrixTree(matrix_tree->right(), shell_ptrs, density);
  
  matrix_tree->right()->left()->Print();
  matrix_tree->right()->right()->Print();
  
  
  return 0;
  
}