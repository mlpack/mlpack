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


void SchwartzPrescreening::ComputeFockMatrix(Matrix* fock_out, 
                                             Matrix* coulomb_out, 
                                             Matrix* exchange_out) {

  fx_timer_start(module_, "prescreening_time");

  printf("====Screening Shell Pairs====\n");
  fx_timer_start(module_, "shell_screening_time");
  num_shell_pairs_ = eri::ComputeShellPairs(&shell_pair_list_, basis_list_, 
                                            shell_pair_threshold_);
  fx_timer_stop(module_, "shell_screening_time");
  
  fx_result_int(module_, "num_shell_pairs", num_shell_pairs_);
  fx_result_int(module_, "num_shell_pairs_screened", 
                num_shells_ * ((num_shells_ - 1) / 2) - num_shell_pairs_ + num_shells_);
  
  
  printf("====Screening and Computing Integrals====\n");
  
  fx_timer_start(module_, "integral_time");
  for (index_t i = 0; i < num_shell_pairs_; i++) {
  
    ShellPair& i_pair = shell_pair_list_[i];
  
    for (index_t j = 0; j < num_shell_pairs_; j++) {
    
      ShellPair& j_pair = shell_pair_list_[j];

      // consider all the relevant entries here 
      double density_bound = 1.0;
    
      double this_est = i_pair.schwartz_factor() * 
          j_pair.schwartz_factor() * density_bound;
          
      
      if (this_est > threshold_) {
      
        double integral = eri::ComputeShellIntegrals(i_pair, j_pair);
        
        //printf("this_int: %g\n", this_int);
        
        // extend this for general case
        index_t i_ind = i_pair.M_index();
        index_t j_ind = i_pair.N_index();
        
        index_t k_ind = j_pair.M_index();
        index_t l_ind = j_pair.N_index();
        
        double coulomb_int = integral * density_matrix_.ref(k_ind, l_ind);
        
        double coulomb_val = coulomb_matrix_.get(i_ind, j_ind);
        
        double exchange_ik = density_matrix_.ref(j_ind, l_ind) * integral;
        double exchange_il = density_matrix_.ref(j_ind, k_ind) * integral;
        double exchange_jk = density_matrix_.ref(i_ind, l_ind) * integral;
        double exchange_jl = density_matrix_.ref(i_ind, k_ind) * integral;
        
        if (k_ind != l_ind) {
          coulomb_int *= 2;
        }
          
        coulomb_val += coulomb_int;
        
        coulomb_matrix_.set(i_ind, j_ind, coulomb_val); 
        coulomb_matrix_.set(j_ind, i_ind, coulomb_val);
        
        exchange_matrix_.set(i_ind, k_ind, 
                          exchange_matrix_.ref(i_ind,k_ind) + exchange_ik);
        if (k_ind != l_ind) {
          exchange_matrix_.set(i_ind, l_ind, 
                            exchange_matrix_.ref(i_ind,l_ind) + exchange_il);
        }
        if (i_ind != j_ind) {
          exchange_matrix_.set(j_ind, k_ind, 
                            exchange_matrix_.ref(j_ind,k_ind) + exchange_jk);
        }
        if ((k_ind != l_ind) && (i_ind != j_ind)) {
          exchange_matrix_.set(j_ind, l_ind, 
                            exchange_matrix_.ref(j_ind,l_ind) + exchange_jl);
        }
        
      } 
      else {
        num_prunes_++;
      }
    
    }
    
  } // for i
    
  // F = J - 1/2 K
  la::Scale(0.5, &exchange_matrix_);
  la::SubInit(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  
  fx_timer_stop(module_, "integral_time");
  
  fx_timer_stop(module_, "prescreening_time");
    
  fock_out->Copy(fock_matrix_);
  coulomb_out->Copy(coulomb_matrix_);
  exchange_out->Copy(exchange_matrix_);
  
  fx_result_int(module_, "num_prunes", num_prunes_);

//  printf("num_prunes: %d\n", num_prunes_);

} // ComputeFockMatrix()





