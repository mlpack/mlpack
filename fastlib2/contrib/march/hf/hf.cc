/**
 * @file hf.cc
 * 
 * @author Bill March (march@gatech.edu)
 *
 * Has the main function for a basic Hartree-Fock calculation.
 */

#include "hf.h"

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv);
  
  const double angstroms_to_bohr = 1.889725989;
  
  struct datanode* mod = fx_submodule(NULL, "hf", "hf");
  
  int num_electrons = fx_param_int_req(NULL, "num_electrons");
  
  const char* centers_file = fx_param_str(NULL, "basis_centers", 
                                          "test_centers.csv");
  Matrix centers;
  data::Load(centers_file, &centers);
  
  const char* nuclear_file = fx_param_str(NULL, "nuclear_centers", 
                                          "test_nuclear_centers.csv");
                                          
  Matrix nuclear;
  data::Load(nuclear_file, &nuclear);
  
                                               
  Matrix nuclear_masses;
  
  int single_nuclear_charge = fx_param_int(NULL, "single_charge", 0);
  
  if (single_nuclear_charge > 0) {
  
    nuclear_masses.Init(nuclear.n_cols(), 1);
    nuclear_masses.SetAll(single_nuclear_charge);
  
  }
  else {
  
    const char* nuclear_mass_file = fx_param_str(NULL, "nuclear_masses", 
                                                 "test_nuclear_masses.csv");
    
    data::Load(nuclear_mass_file, &nuclear_masses);
  
  }
  
  // Need to double check if this is right
  if (nuclear.n_cols() != nuclear_masses.n_rows()) {
    FATAL("Number of masses must equal number of nuclear coordinates!\n");
  }
  
  Vector nuclear_mass;
  nuclear_masses.MakeColumnVector(0, &nuclear_mass);
  
  if (fx_param_exists(NULL, "angstroms")) {
  
    la::Scale(angstroms_to_bohr, &centers);
    la::Scale(angstroms_to_bohr, &nuclear);
  
  }
  
  Matrix density;
  if (fx_param_exists(NULL, "initial_density")) {
    const char* density_file = fx_param_str_req(NULL, "initial_density");
    data::Load(density_file, &density);
  }
  else {
    density.Init(centers.n_cols(), centers.n_cols());
    density.SetZero();
  }
  
  
  SCFSolver solver;
  
  solver.Init(mod, num_electrons, centers, density, nuclear, nuclear_mass);
  
  solver.ComputeWavefunction();
  
  fx_done();
  
  return 0;
  
}
