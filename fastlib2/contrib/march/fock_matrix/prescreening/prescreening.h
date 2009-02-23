/**
 * Headers for integral prescreening
 *
 */

#include "fastlib/fastlib.h"


// Need to determine all shell pairs that meet the cutoff


class IntegralPrescreening {

 public:

  IntegralPrescreening() {}
  
  ~IntegralPrescreening() {}
  
  
  
  
 private:
  
  // The value at which to not count the shell pair
  double bound_cutoff_;
  
  // The vector of significant shell pairs
  // Do I compute the new centers and bandwidths now?
  // I think this is necessary for the CFMM but I'm not sure for LinK/ONX
  Vector shell_pairs_;
  
  index_t num_orbitals_;
  
  index_t num_shell_pairs_;
  
  
  /**
   * Returns the computed bound for the two AOs
   */
  double compute_bound_(index_t i, index_t j);
  
  
  
  
 public:
  
    
  /**
   * Main routine, fills vector of significant shell pairs
   */
  void compute_shell_pairs();

}; // class IntegralPrescreening
