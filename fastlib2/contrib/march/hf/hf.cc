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
  
  
  
  
  ////////////// Compute the integrals ///////////
  
  // Should this be in the same, or a different class from the linear system
  // solver?
  
  
  
  ////////////// Solve the linear system /////////////
  
  
  
  
  
  
  //////////// Output the results ///////////////////
  
  
  
  
  fx_done();
  
  return 0;
  
}