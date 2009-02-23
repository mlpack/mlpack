#include "multi_tree_fock.h"


bool CanApproximateCoulomb_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma, 
                            double* approx_val) {

  

}


bool CanApproximateExchange_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma, 
                             double* approx_val) {
    
  
  
}

void ComputeCoulombBaseCase_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma) {
                             
                             
                             
                             
}


void ComputeExchangeBaseCase_(SquareFockTree* mu_nu, 
                              SquareFockTree* rho_sigma) {
 
                              
}

void ComputeFockMatrix() {

  fx_timer_start(module_, "coulomb_recursion");
  ComputeCoulombRecursion_(square_tree_, square_tree_);  
  fx_timer_stop(module_, "coulomb_recursion");

  // Will need to be followed by clearing the tree and computing the exchange 
  // matrix
  // I think this is the only resetting the tree will need
  SetEntryBounds_();
  ResetTreeForExchange_(square_tree_);
  fx_timer_start(module_, "exchange_recursion");
  ComputeExchangeRecursion_(square_tree_, square_tree_);
  fx_timer_stop(module_, "exchange_recursion");
    
  la::SubOverwrite(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  

} // ComputeFockMatrix()