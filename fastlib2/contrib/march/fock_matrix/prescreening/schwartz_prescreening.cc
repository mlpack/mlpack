/**
 * Prescreening with Schwartz bound
 */

#include "schwartz_prescreening.h"


// maybe this should go in eri?
// Needs to be updated for p-type
/*
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
 */


// maybe this should go in ERI? 
/*
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
*/

void SchwartzPrescreening::UpdateDensity(const Matrix& new_density) {
 
  density_matrix_.CopyValues(new_density);
  
  coulomb_matrix_.SetZero();
  exchange_matrix_.SetZero();
  
  /*
  if (!first_computation_) {
    shell_pair_list_.Clear();
  }
  */
  
  //num_prunes_ = 0;
  
}


void SchwartzPrescreening::Compute() {

  printf("====Screening and Computing Integrals====\n");
  
  fx_timer_start(module_, "prescreening_time");
  
  fx_timer_start(module_, "integral_time");
  for (index_t a = 0; a < num_shell_pairs_; a++) {
  
    ShellPair& A_pair = shell_pair_list_[a];
    
    Matrix coulomb_ij;
    coulomb_ij.Init(A_pair.M_Shell()->num_functions(), 
                    A_pair.N_Shell()->num_functions());
    coulomb_ij.SetZero();
  
    for (index_t b = 0; b < num_shell_pairs_; b++) {
    
      ShellPair& B_pair = shell_pair_list_[b];

      // consider all the relevant entries here 
      // I think these need to be in absolute value
      
      /*
      double density_bound = max(A_pair.density_bound(), B_pair.density_bound());
      printf("density_bound: %g\n", density_bound);
      
      for (index_t k_ind = 0; k_ind < B_pair.M_Shell()->num_functions(); k_ind++) {
        
        for (index_t i_ind = 0; i_ind < A_pair.M_Shell()->num_functions(); i_ind++) {
          
          density_bound = max(density_bound, 
                              0.25 * fabs(density_matrix_.ref(k_ind, i_ind)));
          
        } // i_ind
        
        for (index_t j_ind = 0; j_ind < A_pair.N_Shell()->num_functions(); j_ind++) {
          
          density_bound = max(density_bound, 
                              0.25 * fabs(density_matrix_.ref(k_ind, j_ind)));
          
        } // j_ind
        
      } // for k_ind
      
      // should wrap this in a check to see if k and l are different to save time
      for (index_t l_ind = 0; l_ind < B_pair.N_Shell()->num_functions(); l_ind++) {
        
        for (index_t i_ind = 0; i_ind < A_pair.M_Shell()->num_functions(); i_ind++) {
          
          density_bound = max(density_bound, 
                              0.25 * fabs(density_matrix_.ref(l_ind, i_ind)));
          
        } // i_ind
        
        for (index_t j_ind = 0; j_ind < A_pair.N_Shell()->num_functions(); j_ind++) {
          
          density_bound = max(density_bound, 
                              0.25 * fabs(density_matrix_.ref(l_ind, j_ind)));
          
        } // j_ind
        
      } // for l_ind
       */
      
      double density_bound = eri::DensityBound(A_pair, B_pair, density_matrix_);
      
      //printf("density_bound: %g\n", density_bound);
      
      double this_est = A_pair.schwartz_factor() * 
          B_pair.schwartz_factor() * density_bound;
      //printf("this_est: %g\n", this_est);
          
      
      if (this_est > threshold_) {
      
        // now computing and contracting the integral
        IntegralTensor integrals;
        eri::ComputeShellIntegrals(A_pair, B_pair, &integrals);
        //integrals.Print();
        //printf("\n");
        num_integrals_computed_++;
        //printf("this_int: %g\n", this_int);
        
        
        integrals.ContractCoulomb(B_pair.M_Shell()->matrix_indices(),
                                  B_pair.N_Shell()->matrix_indices(),
                                  density_matrix_, &coulomb_ij, 
                                  (B_pair.M_Shell() == B_pair.N_Shell()));
        //printf("Coulomb integral: (%d, %d, %d, %d): %g\n", i, j, k, l, 
        //       integrals.ref(0, 0, 0, 0) * density_.ref(k, l));
        
        Matrix exchange_ik;
        exchange_ik.Init(A_pair.M_Shell()->num_functions(), 
                         B_pair.M_Shell()->num_functions());
        exchange_ik.SetZero();
        
        Matrix* exchange_jk;
        Matrix* exchange_il;
        Matrix* exchange_jl;
        
        // not sure if this will work with the references
        // if i != j
        if (A_pair.M_Shell() != A_pair.N_Shell()) {
          exchange_jk = new Matrix();
          exchange_jk->Init(A_pair.N_Shell()->num_functions(), 
                            B_pair.M_Shell()->num_functions());
          exchange_jk->SetZero();
        }
        else {
          exchange_jk = NULL;
        }
        
        // if k != l
        if (B_pair.N_Shell() != B_pair.M_Shell()) {
          exchange_il = new Matrix();
          exchange_il->Init(A_pair.M_Shell()->num_functions(), 
                            B_pair.N_Shell()->num_functions());
          exchange_il->SetZero();
          
        }
        else {
          exchange_il = NULL;
        }
        
        if (A_pair.M_Shell() != A_pair.N_Shell() && 
            B_pair.N_Shell() != B_pair.M_Shell()) {
          exchange_jl = new Matrix();
          exchange_jl->Init(A_pair.N_Shell()->num_functions(), 
                            B_pair.N_Shell()->num_functions());
          exchange_jl->SetZero();
          
        }
        else {
          exchange_jl = NULL; 
        }
        
        integrals.ContractExchange(A_pair.M_Shell()->matrix_indices(),
                                   A_pair.N_Shell()->matrix_indices(),
                                   B_pair.M_Shell()->matrix_indices(),
                                   B_pair.N_Shell()->matrix_indices(),
                                   density_matrix_, &exchange_ik, exchange_jk, 
                                   exchange_il, exchange_jl);
        
        eri::AddSubmatrix(A_pair.M_Shell()->matrix_indices(), 
                          B_pair.M_Shell()->matrix_indices(),
                          exchange_ik, &exchange_matrix_);
        
        if (exchange_jk) {
          eri::AddSubmatrix(A_pair.N_Shell()->matrix_indices(), 
                            B_pair.M_Shell()->matrix_indices(),
                            *exchange_jk, &exchange_matrix_);
	  delete exchange_jk;
	}
        
        if (exchange_il) {
          eri::AddSubmatrix(A_pair.M_Shell()->matrix_indices(), 
                            B_pair.N_Shell()->matrix_indices(),
                            *exchange_il, &exchange_matrix_);
	  delete exchange_il;
	}
        
        if (exchange_jl) {
          eri::AddSubmatrix(A_pair.N_Shell()->matrix_indices(), 
                            B_pair.N_Shell()->matrix_indices(),
                            *exchange_jl, &exchange_matrix_);
	  delete exchange_jl;
	}
        
      } // estimate meets the threshold
      else {
        num_prunes_++;
      }
    
    } // for b
    
    // now add in the coulomb integrals
    eri::AddSubmatrix(A_pair.M_Shell()->matrix_indices(), 
                      A_pair.N_Shell()->matrix_indices(),
                      coulomb_ij, &coulomb_matrix_);
    
    if (A_pair.M_Shell() != A_pair.N_Shell()) {
      
      Matrix coulomb_ji;
      la::TransposeInit(coulomb_ij, &coulomb_ji);
      eri::AddSubmatrix(A_pair.N_Shell()->matrix_indices(), 
                        A_pair.M_Shell()->matrix_indices(),
                        coulomb_ji, &coulomb_matrix_);
    }
    
    
  } // for a
    
  // F = J - 1/2 K
  la::Scale(0.5, &exchange_matrix_);
  la::SubOverwrite(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  
  fx_timer_stop(module_, "integral_time");
  
  fx_timer_stop(module_, "prescreening_time");
    
  fx_result_int(module_, "num_prunes", num_prunes_);
  
  fx_result_int(module_, "num_integrals_computed", num_integrals_computed_);

  first_computation_ = false;


} // Compute()


void SchwartzPrescreening::OutputFock(Matrix* fock_out, Matrix* coulomb_out, 
                                      Matrix* exchange_out) {
  
  if (fock_out) {
    fock_out->Copy(fock_matrix_);
  }
  if (coulomb_out) {
    coulomb_out->Copy(coulomb_matrix_);
  }
  if (exchange_out) {
    exchange_out->Copy(exchange_matrix_);
  }
  
} // OutputFock

void SchwartzPrescreening::OutputCoulomb(Matrix* coulomb_out) {
  if (coulomb_out) {
    coulomb_out->Copy(coulomb_matrix_);
  }  
}

void SchwartzPrescreening::OutputExchange(Matrix* exchange_out) {
  if (exchange_out) {
    exchange_out->Copy(exchange_matrix_);
  }
}





