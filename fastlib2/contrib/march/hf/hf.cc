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
  
  ////////////// Read in data //////////////
  
  // How will the data be organized?
  // What is the best format to read in basis functions?
  // I will likely need my own function to parse basis functions
  // Check out the PSI3 code 
  
  
  
  ////////////// Compute the integrals ///////////
  
  // Should this be in the same, or a different class from the linear system
  // solver?
  
  Matrix fock_matrix;
  Matrix overlap_matrix;
  
  ////////////// Solve the linear system /////////////
  
  //HFSolver solver;
  //solver.Init(fock_matrix, overlap_matrix);
  
  
  
  
  //////////// Output the results ///////////////////
  
  // Total energy
  // Spin orbitals: both filled and virtual
  
  
  fx_done();
  
  return 0;
  
}