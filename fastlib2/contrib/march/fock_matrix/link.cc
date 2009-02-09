#include "link.h"



void Link::ComputeFockMatrix() {

  // Find "significant" bra and ket shell pairs?
  
  // When I do this, I need to also create a list of the maximum Schwartz 
  // integral estimate for each shell
  num_shell_pairs_ = eri::ComputeShellPairs(&shell_pair_list_, shell_list_, 
                                            shell_pair_cutoff_, &shell_max_);
  
  // Find significant \nu for all \mu
  
  // I am supposed to loop over all mu in significant bra shell pairs and over 
  // all nu in significant ket shell pairs
  // Maybe I should just loop over the shells themselves and assume that each
  // shell appears in at least one significant shell pair
  for (index_t i = 0; i < num_shells_; i++) {
  
    significant_nu_for_mu = significant_mu_pairs_[i];
    significant_nu_for_mu.Init(num_shells_/2);
    next_ind = 0;
        
    for (index_t j = 0; j < num_shells_; j++) {
    
      // I think this threshold should be the same as the others 
      // Not sure if this is the right density matrix entry
      if (density_mat_.ref(i,j) * shell_max_[i] * shell_max_[j] > threshold_) {
        
        // Store significant j for each i
        // What is the best way to do this?  This list will have to be sorted
        // This fails if there are more than num_shells_/2 significant elements
        significant_nu_for_mu[next_ind] = j;
        next_ind++;
      
      }
    
    } // for j
    
    // sort significant_nu_for_mu
    
  
  } // for i
  
  // loop over bra shell pairs
  
  
  
  // loop over \nu for the current \mu
  
  // loop over \sigma and screen to see if this integral is necessary
  
  // loop over \lambda and screen
  
  // compute significant integrals

} // ComputeFockMatrix()



void Link::OutputFockMatrix(Matrix* exc_out) {

  exc_out->Copy(exchange_matrix_);
  
} // OutputFockMatrix