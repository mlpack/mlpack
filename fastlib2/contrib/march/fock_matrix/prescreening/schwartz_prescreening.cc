/**
 * Prescreening with Schwartz bound
 */

#include "schwartz_prescreening.h"


// maybe this should go in eri?
// Needs to be updated for p-type
double SchwartzPrescreening::ComputeSchwartzIntegral_(BasisShell& mu, 
                                                      BasisShell& nu) {
  
  Vector& cent1 = mu.center();
  Vector& cent2 = nu.center();
  double exp1 = mu.exp();
  double exp2 = nu.exp();
  
  double this_int = eri::SSSSIntegral(exp1, cent1, exp2, cent2, 
                                      exp1, cent1, exp2, cent2);
  
  this_int = this_int * mu.normalization_constant() * mu.normalization_constant();
  this_int = this_int * nu.normalization_constant() * nu.normalization_constant();
  
  return this_int;
  
} // ComputeSchwartzIntegral_()


// maybe this should go in ERI? 
double SchwartzPrescreening::SchwartzBound_(BasisShell &mu, 
                                             BasisShell &nu) {
                                 
  index_t i_funs = mu.num_functions();
  index_t j_funs = nu.num_functions();
                                                                  
  double Q_mu_nu = -DBL_MAX;
  for (index_t i = 0; i < i_funs; i++) {
  
    for (index_t j = 0; j < j_funs; j++) {
      
      double this_Q = ComputeSchwartzIntegral_(mu, nu);
      
      if (this_Q > Q_mu_nu) {
        
        Q_mu_nu = this_Q;
        
      }
    
    }
  
  }
  
  return sqrt(Q_mu_nu);

} // SchwartzBound_()


void SchwartzPrescreening::ComputeFockMatrix(Matrix* fock_out) {

  // form shell-pairs
  /*
  for (index_t i = 0; i < num_shells_; i++) {
  
    BasisShell i_shell = basis_list_[i];
  
    for (index_t j = i; j < num_shells_; j++) {
    
      BasisShell j_shell = basis_list_[j];
      
      shell_pair_list_[num_shell_pairs_].Init(i, j, i_shell, j_shell);
      
      double this_bound = SchwartzBound_(i_shell, j_shell);
      //printf("this_bound = %g\n", this_bound);
      shell_pair_list_[num_shell_pairs_].set_integral_upper_bound(this_bound);
      // set density bound too
      
      num_shell_pairs_++;
      
      
    } // for j
  
  } //for i
  */
  num_shell_pairs_ = eri::ComputeShellPairs(&shell_pair_list_, basis_list_, 
                                            shell_pair_threshold_);
  
  fx_format_result(module_, "num_shell_pairs", "%d", num_shell_pairs_);
  
  // Is there symmetry I can use here?
  for (index_t i = 0; i < num_shell_pairs_; i++) {
  
    ShellPair& i_pair = shell_pair_list_[i];
  
    for (index_t j = 0; j < num_shell_pairs_; j++) {
    
      ShellPair& j_pair = shell_pair_list_[j];

      // this can be made tighter by considering other density matrix entries
      //double density_bound = max(i_pair.density_mat_upper(), 
      //                           j_pair.density_mat_upper());
      double density_bound = 1.0;
    
      /*printf("i upper = %g\n", i_pair.integral_upper_bound());
      printf("j upper = %g\n", j_pair.integral_upper_bound());
      */
      double this_est = i_pair.integral_upper_bound() * 
          j_pair.integral_upper_bound() * density_bound;
          
      //printf("this_est = %g\n", this_est);
      //printf("thresh = %g\n\n", threshold_);
      
      
      if (this_est > threshold_) {
      
        double this_int = eri::ComputeShellIntegrals(i_pair, j_pair);
        
        //printf("this_int: %g\n", this_int);
        
        // extend this for general case
        index_t mind = i_pair.M_index();
        index_t nind = i_pair.N_index();
        
        index_t kind = j_pair.M_index();
        index_t lind = j_pair.N_index();
        
        //printf("mind: %d\n", mind);
        //printf("nind: %d\n", nind);
        
        // don't forget symmetry
        double cval = coulomb_matrix_.get(mind, nind);
        double eval = exchange_matrix_.get(mind, nind);
        
        if (kind != lind) {
          cval = cval + 2 * this_int;
          eval = eval - this_int;
        }
        else {
          cval = cval + this_int;
          eval = eval - 0.5*this_int;
        }
        
        
        coulomb_matrix_.set(mind, nind, cval); 
        exchange_matrix_.set(mind, nind, eval);
        
        if (mind != nind) {
          coulomb_matrix_.set(nind, mind, cval);
          exchange_matrix_.set(nind, mind, eval);
        }
              
      } 
      else {
        num_prunes_++;
      }
    
    }
    
  } // for i
  
  la::AddInit(coulomb_matrix_, exchange_matrix_, &fock_matrix_);
  
  fock_out->Copy(fock_matrix_);
  
  fx_format_result(module_, "num_prunes", "%d", num_prunes_);

//  printf("num_prunes: %d\n", num_prunes_);

} // ComputeFockMatrix()





