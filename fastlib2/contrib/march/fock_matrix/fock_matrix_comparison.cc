#include "fock_matrix_comparison.h"

void FockMatrixComparison::Init(fx_module* exp_mod, Matrix** exp_mats, 
                                fx_module* naive_mod, Matrix** naive_mats, 
                                fx_module* my_mod) {
           
  approx_mod_ = exp_mod;
  naive_mod_ = naive_mod;
  
  F_mat_ = exp_mats[0];
  J_mat_ = exp_mats[1];
  K_mat_ = exp_mats[2];
  
  naive_F_mat_ = naive_mats[0];
  naive_J_mat_ = naive_mats[1];
  naive_K_mat_ = naive_mats[2];
  
  compare_fock_ = F_mat_ != NULL;
  compare_coulomb_ = J_mat_ != NULL;
  compare_exchange_ = K_mat_ != NULL;
  
  my_mod_ = my_mod;
  
  max_diff_F_ = -1.0;
  max_diff_J_ = -1.0;
  max_diff_K_ = -1.0;
  
  max_rel_F_ = -1.0;
  max_rel_J_ = -1.0;
  max_rel_K_ = -1.0;
  
  total_diff_F_ = 0.0;
  total_diff_J_ = 0.0;
  total_diff_K_ = 0.0;
  
  num_entries_ = naive_F_mat_->n_cols();
  
  rel_error_cutoff_ = 10e-30;
  
  // check that all matrices are the same size
  
} // Init()


void FockMatrixComparison::Compare() {

  printf("====== Comparing Matrices ======\n");

  for (index_t i = 0; i < num_entries_; i++) {
  
    for (index_t j = i; j < num_entries_; j++) {
    
      if (compare_fock_) {
      
        double F_entry1 = F_mat_->ref(i, j);
        double F_entry2 = naive_F_mat_->ref(i, j);
      
        double this_diff = fabs(F_entry1 - F_entry2);
        double this_rel_diff = fabs(this_diff/F_entry2);
            
        if (this_diff > max_diff_F_) {
          max_diff_F_ = this_diff;
        }
        
        // add in the relative error cutoff to keep from storing large 
        // relative errors from very small sources
        if ((this_rel_diff > max_rel_F_) && (this_diff > rel_error_cutoff_)) {
          max_rel_F_ = this_rel_diff;
          max_index_row_F_ = i;
          max_index_col_F_ = j;
        }
        
        // account for symmetry
        if (i != j) {
          this_diff *= 2;
        }
        
        total_diff_F_ += this_diff;
            
      } // compare_fock
      
      if (compare_coulomb_) {
      
        double J_entry1 = J_mat_->ref(i, j);
        double J_entry2 = naive_J_mat_->ref(i, j);
        
        double this_diff = fabs(J_entry1 - J_entry2);
        double this_rel_diff = fabs(this_diff/J_entry2);
        
        if (this_diff > max_diff_J_) {
          max_diff_J_ = this_diff;
          
        }
        
        if ((this_rel_diff > max_rel_J_) && (this_diff > rel_error_cutoff_)) {
          max_rel_J_ = this_rel_diff;
          max_index_row_J_ = i;
          max_index_col_J_ = j;
        }
        
        // account for symmetry
        if (i != j) {
          this_diff *= 2;
        }
        
        total_diff_J_ += this_diff;        
      
      } // compare coulomb
      
      if (compare_exchange_) {
        
        double K_entry1 = K_mat_->ref(i, j);
        double K_entry2 = naive_K_mat_->ref(i, j);
        
        double this_diff = fabs(K_entry1 - K_entry2);
        double this_rel_diff = fabs(this_diff/K_entry2);
        
        if (this_diff > max_diff_K_) {
          max_diff_K_ = this_diff;
        }
        
        if ((this_rel_diff > max_rel_K_) && (this_diff > rel_error_cutoff_)) {
          max_rel_K_ = this_rel_diff;
          max_index_row_K_ = i;
          max_index_col_K_ = j;
        }
        
        // account for symmetry
        if (i != j) {
          this_diff *= 2;
        }
        
        total_diff_K_ += this_diff;
        
      } // compare exchange
      
    } // for j
  
  } // for i
  
  
  // compute eigenvalue differences
  
  if (compare_fock_) {
    
    printf("Can't compare eigenvalues yet, add core Hamiltonian code.\n");
    
  } // compare fock eigenvalues
  
  // output results 
  
  if (compare_fock_) {
    fx_result_double(my_mod_, "max_diff_F", max_diff_F_);
    fx_result_int(my_mod_, "max_index_row_F", max_index_row_F_);
    fx_result_int(my_mod_, "max_index_col_F", max_index_col_F_);
    fx_result_double(my_mod_, "ave_diff_F", 
                     (total_diff_F_/(num_entries_ * num_entries_)));
    fx_result_double(my_mod_, "max_rel_diff_F", max_rel_F_);
  }

  if (compare_coulomb_) {
    fx_result_double(my_mod_, "max_diff_J", max_diff_J_);
    fx_result_int(my_mod_, "max_index_row_J", max_index_row_J_);
    fx_result_int(my_mod_, "max_index_col_J", max_index_col_J_);
    fx_result_double(my_mod_, "ave_diff_J", 
                     (total_diff_J_/(num_entries_ * num_entries_)));
    fx_result_double(my_mod_, "max_rel_diff_J", max_rel_J_);
  }

  if (compare_exchange_) {
    fx_result_double(my_mod_, "max_diff_K", max_diff_K_);
    fx_result_int(my_mod_, "max_index_row_K", max_index_row_K_);
    fx_result_int(my_mod_, "max_index_col_K", max_index_col_K_);
    fx_result_double(my_mod_, "ave_diff_K", 
                     (total_diff_K_/(num_entries_ * num_entries_)));
    fx_result_double(my_mod_, "max_rel_diff_K", max_rel_K_);
  }
                   
} // Compare()

// not sure this class actually owns these matrices
void FockMatrixComparison::Destruct() {

  if (F_mat_) {
    F_mat_->Destruct();
  }
  if (naive_F_mat_) {
    naive_F_mat_->Destruct();
  }
  if (J_mat_) {
    J_mat_->Destruct();
  }
  if (naive_J_mat_) {
    naive_J_mat_->Destruct();
  }
  if (K_mat_) {
    K_mat_->Destruct();
  }
  if (naive_K_mat_) {
    naive_K_mat_->Destruct();
  }
  
  my_mod_ = NULL;
  approx_mod_ = NULL;
  naive_mod_ = NULL;
  
  
} // Destruct()




