/*
 *  naive_fock_main.cc
 *  
 *
 *  Created by William March on 8/4/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
 *
 */


#include "naive_fock_matrix.h"
#include "contrib/march/fock_matrix/chem_reader/chem_reader.h"
#include "contrib/march/fock_matrix/fock_impl/eri.h"

int main(int argc, char* argv[]) {

  fx_module* root_mod = fx_init(argc, argv, NULL);
  
  Matrix centers;
  const char* centers_file = fx_param_str_req(root_mod, "centers");
  data::Load(centers_file, &centers);
  
  Matrix exp_mat;
  const char* exp_file = fx_param_str_req(root_mod, "exponents");
  data::Load(exp_file, &exp_mat);
  
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
  
  // WARNING: this is a hack for now
  index_t num_funs = centers.n_cols() + 2 * (index_t)la::Dot(momenta, momenta);
  
  
  std::string density_str;
  Matrix density;
  if (fx_param_exists(root_mod, "density")) {
    
    const char* density_file = fx_param_str_req(root_mod, "density");
    density_str = density_file;
    
    size_t density_ext = density_str.find_last_of(".");
    if (!strcmp("qcmat", 
                density_str.substr(density_ext+1, std::string::npos).c_str())) {
      
      printf("Reading QChem Style Density Matrix.\n");
      chem_reader::ReadQChemDensity(density_file, &density, num_funs);
      
    }
    else {
      
      printf("Reading csv Density Matrix.\n");
      data::Load(density_file, &density);
      
    }
  }
  else {
    
    
    density.Init(num_funs, num_funs);
    density.SetAll(1.0);
    density_str = "default";
  }
  
  //density.PrintDebug("Density (from input file)");
  /* NO LONGER TRUE
   if ((density.n_cols() != centers.n_cols()) || 
   (density.n_rows() != centers.n_cols())) {
   FATAL("Density matrix has wrong dimensions.\n");
   }
   */
  
    
  const double angstrom_to_bohr = 1.889725989;
  // if the data are not input in bohr, assume they are in angstroms
  if (!fx_param_exists(root_mod, "bohr")) {
    
    la::Scale(angstrom_to_bohr, &centers);

  }
  
  eri::ERIInit();
  
  Matrix naive_fock;
  Matrix naive_coulomb;
  Matrix naive_exchange;
  
  fx_module* naive_mod = fx_submodule(root_mod, "naive");
  
  NaiveFockMatrix naive_alg;
  naive_alg.Init(centers, exp_mat, momenta, density, naive_mod);
  
  naive_alg.Compute();
  naive_alg.OutputFock(&naive_fock, &naive_coulomb, 
                       &naive_exchange);
  
  if (fx_param_exists(root_mod, "print_naive")) {
    
    naive_fock.PrintDebug("Naive F");
    naive_coulomb.PrintDebug("Naive J");
    naive_exchange.PrintDebug("Naive K");
    
  }
  
  eri::ERIFree();
  
  return 0;
  
} // main