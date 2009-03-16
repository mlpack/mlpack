#include "fock_matrix_comparison.h"

void FockMatrixComparison::Init(fx_module* mod1, Matrix** mat1, 
                                fx_module* mod2, Matrix** mat2, 
                                fx_module* my_mod) {
           
  mod1_ = mod1;
  mod2_ = mod2;
  
  mat1_F_ = mat1[0];
  mat1_J_ = mat1[1];
  mat1_K_ = mat1[2];
  
  mat2_F_ = mat2[0];
  mat2_J_ = mat2[1];
  mat2_K_ = mat2[2];
  
  compare_fock_ = (mat1_F_ != NULL) && (mat2_F_ != NULL);
  compare_coulomb_ = (mat1_J_ != NULL) && (mat2_J_ != NULL);
  compare_exchange_ = (mat1_K_ != NULL) && (mat2_K_ != NULL);
  
  my_mod_ = my_mod;
  
  max_diff_F_ = 0.0;
  max_diff_J_ = 0.0;
  max_diff_K_ = 0.0;
  
  total_diff_F_ = 0.0;
  total_diff_J_ = 0.0;
  total_diff_K_ = 0.0;
  
  num_entries_ = mat1_F_->n_cols();
  
  // check that all matrices are square
  DEBUG_ASSERT(mat1_F_->n_cols() == mat1_F_->n_rows());
  DEBUG_ASSERT(mat2_F_->n_cols() == mat2_F_->n_rows());
  
  // check that all matrices are the same size
  DEBUG_ASSERT(mat1_F_->n_cols() == mat2_F_->n_cols());
  
  
  
                                                     
} // Init()


void FockMatrixComparison::Compare() {

  for (index_t i = 0; i < num_entries_; i++) {
  
    for (index_t j = i; j < num_entries_; j++) {
    
      if (compare_fock_) {
      
        double F_entry1 = mat1_F_->ref(i, j);
        double F_entry2 = mat2_F_->ref(i, j);
      
        double this_diff = abs(F_entry1 - F_entry2);
            
        if (this_diff > max_diff_F_) {
          max_diff_F_ = this_diff;
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
      
        double J_entry1 = mat1_J_->ref(i, j);
        double J_entry2 = mat2_J_->ref(i, j);
        
        double this_diff = abs(J_entry1 - J_entry2);
        
        if (this_diff > max_diff_J_) {
          max_diff_J_ = this_diff;
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
        
        double K_entry1 = mat1_K_->ref(i, j);
        double K_entry2 = mat2_K_->ref(i, j);
        
        double this_diff = abs(K_entry1 - K_entry2);
        
        if (this_diff > max_diff_K_) {
          max_diff_K_ = this_diff;
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
  
  fx_result_double(my_mod_, "max_diff_F", max_diff_F_);
  fx_result_int(my_mod_, "max_index_row_F", max_index_row_F_);
  fx_result_int(my_mod_, "max_index_col_F", max_index_col_F_);
  fx_result_double(my_mod_, "ave_diff_F", 
                   (total_diff_F_/(num_entries_ * num_entries_)));

  fx_result_double(my_mod_, "max_diff_J", max_diff_J_);
  fx_result_int(my_mod_, "max_index_row_J", max_index_row_J_);
  fx_result_int(my_mod_, "max_index_col_J", max_index_col_J_);
  fx_result_double(my_mod_, "ave_diff_J", 
                   (total_diff_J_/(num_entries_ * num_entries_)));

  fx_result_double(my_mod_, "max_diff_K", max_diff_K_);
  fx_result_int(my_mod_, "max_index_row_K", max_index_row_K_);
  fx_result_int(my_mod_, "max_index_col_K", max_index_col_K_);
  fx_result_double(my_mod_, "ave_diff_K", 
                   (total_diff_K_/(num_entries_ * num_entries_)));

                   
} // Compare()


void FockMatrixComparison::Destruct() {



} // Destruct()




