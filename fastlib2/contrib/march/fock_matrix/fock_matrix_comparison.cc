#include "fock_matrix_comparison.h"

void FockMatrixComparison::Init(fx_module* mod1, const Matrix& mat1, 
                                fx_module* mod2, const Matrix& mat2, 
                                fx_module* my_mod) {
           
  mod1_ = mod1;
  mod2_ = mod2;
  mat1_.Copy(mat1);
  mat2_.Copy(mat2);
  
  my_mod_ = my_mod;
  
  max_diff_ = 0.0;
  total_diff_ = 0.0;
  
  num_entries_ = mat1_.n_cols();
  
  DEBUG_ASSERT(mat1_.n_cols() == mat1_.n_rows());
  DEBUG_ASSERT(mat2_.n_cols() == mat2_.n_rows());
  
  DEBUG_ASSERT(mat1_.n_cols() == mat2_.n_cols());
  
  
  
                                                     
} // Init()


void FockMatrixComparison::Compare() {

  for (index_t i = 0; i < num_entries_; i++) {
  
    for (index_t j = i; j < num_entries_; j++) {
    
      double entry1 = mat1_.ref(i, j);
      double entry2 = mat2_.ref(i, j);
      
      double this_diff = abs(entry1 - entry2);
      
      if (this_diff > max_diff_) {
      
        max_diff_ = this_diff;
        max_index_row_ = i;
        max_index_col_ = j;
      
      } // if
      
      if (i != j) {
        this_diff = 2 * this_diff;
      }
      
      total_diff_ = total_diff_ + this_diff;
      
    
    } // for j
  
  } // for i
  
  // output results 
  
  fx_result_double(my_mod_, "max_diff", max_diff_);
  fx_result_int(my_mod_, "max_index_row", max_index_row_);
  fx_result_int(my_mod_, "max_index_col", max_index_col_);
  fx_result_double(my_mod_, "ave_diff", 
                   (total_diff_/(num_entries_ * num_entries_)));
                   
  mat1_.PrintDebug();
  mat2_.PrintDebug();
                   
                   

} // Compare()


void FockMatrixComparison::Destruct() {

  mat1_.Destruct();
  mat2_.Destruct();

} // Destruct()




