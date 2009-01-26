/**
 * Prescreening with Schwartz bound
 */

#include "fastlib/fastlib.h"

class SchwartzPrescreening {

 public:
  
  SchwartzPrescreening() {}
  
  ~SchwartzPrescreening() {}
  
  void ComputeFockMatrix();
  
  
 private:

  // J
  Matrix coulomb_matrix_;
  // K
  Matrix exchange_matrix_;
  // D
  Matrix density_matrix_;

  // List of all basis shells
  ArrayList<BasisShell> basis_list_;
  
  ArrayList<ShellPair> shell_pair_list_;
  
  index_t num_shells_;
  
  // The threshold for ignoring a shell quartet
  double threshold_;

  /**
   * The result needs to be multiplied by a density matrix bound
   *
   * Maybe the inputs should be shells somehow?  
   */
  double SchwartzBound_(const BasisShell &mu, const BasisShell &nu, 
                        const BasisShell &rho, const BasisShell &sigma);
                        
  
  /**
   * Inner computation for Schwartz bound
   */
  double ComputeSchwartzIntegral_();
  
  


}; // class SchwartzPrescreening