#include "multi_tree_fock.h"




/////////// Recursive Code /////////////////////////

void MultiTreeFock::PassBoundsUp_(MatrixTree* query) {
  
  query->set_remaining_epsilon(min(query->left()->remaining_epsilon(),
                                   query->right()->remaining_epsilon()));
  
  DEBUG_ASSERT(query->left()->remaining_references() == query->right()->remaining_references());
  //printf("query->remaining_references(): %d\n", query->remaining_references());
  query->set_remaining_references(max(query->left()->remaining_references(),
                                      query->right()->remaining_references()));
  
  //printf("Passing bounds up for %p\n", query);
  
}

void MultiTreeFock::PassBoundsDown_(MatrixTree* query) {
  
  query->left()->set_remaining_epsilon(query->remaining_epsilon());
  query->right()->set_remaining_epsilon(query->remaining_epsilon());
  
  query->left()->set_remaining_references(query->remaining_references());
  query->right()->set_remaining_references(query->remaining_references());
  
  query->left()->add_coulomb_approx(query->coulomb_approx_val());
  query->left()->add_exchange_approx(query->exchange_approx_val());
  query->right()->add_coulomb_approx(query->coulomb_approx_val());
  query->right()->add_exchange_approx(query->exchange_approx_val());
  // since it's been passed down now
  query->set_coulomb_approx_val(0.0);
  query->set_exchange_approx_val(0.0);
  
}



void MultiTreeFock::ComputeBaseCase(MatrixTree* query, MatrixTree* reference) {
  
  index_t internal_row_index = 0;
  
  for (index_t i = query->row_shells()->begin(); i < query->row_shells()->end();
       i++) {
    
    BasisShell* i_shell = shell_ptr_list_[i];
    
    index_t internal_col_index = 0;
    
    for (index_t j = query->col_shells()->begin(); j < query->col_shells()->end(); 
         j++) {
      
      BasisShell* j_shell = shell_ptr_list_[j];
      
      Matrix coulomb_ij;
      coulomb_ij.Init(i_shell->num_functions(), j_shell->num_functions());
      coulomb_ij.SetZero();
      
      for (index_t k = reference->row_shells()->begin(); 
           k < reference->row_shells()->end(); k++) {

        BasisShell* k_shell = shell_ptr_list_[k];

        for (index_t l = reference->col_shells()->begin(); 
             l < reference->col_shells()->end(); l++) {
          
          BasisShell* l_shell = shell_ptr_list_[l];

          IntegralTensor coulomb_ints;
          IntegralTensor exchange_ints;
          
          eri::ComputeShellIntegrals(*i_shell, *j_shell, *k_shell, *l_shell, 
                                     &coulomb_ints);
          //printf("Coulomb: (%d, %d, %d, %d)\n", i_shell->matrix_index(0),
          //       j_shell->matrix_index(0), k_shell->matrix_index(0), 
          //       l_shell->matrix_index(0));
          num_integrals_computed_++;
          
          // if the reference node is the same, then these shouldn't be multiplied 
          // by 2, because they'll show up normally in the iterations
          // this should handle the reference symmetry
          coulomb_ints.ContractCoulomb(k_shell->matrix_indices(), 
                                       l_shell->matrix_indices(), 
                                       density_matrix_, &coulomb_ij, 
                                       reference->on_diagonal());
          
          eri::ComputeShellIntegrals(*i_shell, *k_shell, *j_shell, *l_shell,
                                     &exchange_ints);
          num_integrals_computed_++;
          
          Matrix exchange_ik; 
          exchange_ik.Init(i_shell->num_functions(), j_shell->num_functions());
          exchange_ik.SetZero();
          exchange_ints.ContractExchange(i_shell->matrix_indices(),
                                         k_shell->matrix_indices(),
                                         j_shell->matrix_indices(),
                                         l_shell->matrix_indices(),
                                         density_matrix_, &exchange_ik,
                                         NULL, NULL, NULL);
          
          if (!(reference->on_diagonal())) {
            
            // handle the exchange reference symmetry explicitly
            IntegralTensor exchange_ints_sym;
            eri::ComputeShellIntegrals(*i_shell, *l_shell, *j_shell, *k_shell,
                                       &exchange_ints_sym);
            num_integrals_computed_++;
            
            Matrix exchange_il;
            exchange_il.Init(i_shell->num_functions(), j_shell->num_functions());
            exchange_il.SetZero();
            exchange_ints_sym.ContractExchange(i_shell->matrix_indices(),
                                               l_shell->matrix_indices(),
                                               j_shell->matrix_indices(),
                                               k_shell->matrix_indices(),
                                               density_matrix_, &exchange_il,
                                               NULL, NULL, NULL);
            
            la::AddTo(exchange_il, &exchange_ik);
            
          } // handling reference symmetry
          
          // factor of 1/2 for K
          la::Scale(0.5, &exchange_ik);
          
          eri::AddSubmatrix(internal_row_index, i_shell->num_functions(),
                            internal_col_index, j_shell->num_functions(),
                            exchange_ik, query->exchange_entries());
          
        } // for l
        
      } // for k
      
      //coulomb_ij.PrintDebug("Coulomb_ij");
      //printf("internal_row_index: %d, num_funs: %d\n", internal_row_index,
      //       i_shell->num_functions());
      //printf("internal_col_index: %d, num_funs: %d\n", internal_col_index,
      //       j_shell->num_functions());
      // sum coulomb_ij into the node's matrix
      eri::AddSubmatrix(internal_row_index, i_shell->num_functions(), 
                        internal_col_index, j_shell->num_functions(),
                        coulomb_ij, query->coulomb_entries());
      
      // the "below diagonal" (ji) entry should be taken care of in the
      // matrix reconstruction code
      
      internal_col_index += j_shell->num_functions();
      
    } // for j
    
    internal_row_index += i_shell->num_functions();
      
  } // for i
  
  query->set_remaining_references(query->remaining_references() 
                                  - reference->num_pairs());
  
} // ComputeBaseCase

bool MultiTreeFock::SplitQuery(MatrixTree* query, MatrixTree* reference) {
  
  if (query->is_leaf()) {
    DEBUG_ASSERT(!(reference->is_leaf()));
    return false;
  }
  else if (reference->is_leaf()) {
    DEBUG_ASSERT(!(query->is_leaf()));
    return true;
  }
  else {
    return (query->row_shells()->count() + query->col_shells()->count() >=
            reference->row_shells()->count() + reference->col_shells()->count()); 
  }
  
} //SplitQuery()

void MultiTreeFock::NodeBounds(MatrixTree* query, MatrixTree* reference,
                               double* max_coulomb, double* max_exchange,
                               double* min_coulomb, double* min_exchange) {
  
  
  // compute these in ERI somehow
  double max_coulomb_integral, min_coulomb_integral;
  double max_exchange_integral, min_exchange_integral;
  
  // make calls to eri
  // or possibly create an eri_bounds namespace to keep things neat
  // choose which function to call based on momenta of the shells
  max_coulomb_integral = eri_bounds::BoundIntegrals(query->row_shells(),
                                                    query->col_shells(),
                                                    reference->row_shells(),
                                                    reference->col_shells());
    
  // TODO: is this right?  What about the reference symmetry?
  // Do I need to compute both sides and take the max?
  // Only for off diagonal queries?
  max_exchange_integral = eri_bounds::BoundIntegrals(query->row_shells(),
                                                     reference->row_shells(),
                                                     query->col_shells(),
                                                     reference->col_shells());
  
  min_coulomb_integral = -1 * max_coulomb_integral;
  min_exchange_integral = -1 * max_exchange_integral;
  
  // factor in the density
  if (reference->density_bounds().hi >= 0.0) {
    
    *max_coulomb = reference->density_bounds().hi * max_coulomb_integral;
    *max_exchange = reference->density_bounds().hi * max_exchange_integral;
    
  }
  else {
    
    *max_coulomb = reference->density_bounds().hi * min_coulomb_integral;
    *max_exchange = reference->density_bounds().hi * min_exchange_integral;
    
  }
  
  if (reference->density_bounds().lo >= 0.0) {
    
    *min_coulomb = reference->density_bounds().lo * min_coulomb_integral;
    *min_exchange = reference->density_bounds().lo * min_exchange_integral;
    
  }
  else {

    *min_coulomb = reference->density_bounds().lo * max_coulomb_integral;
    *min_exchange = reference->density_bounds().lo * max_exchange_integral;

  }
  
  DEBUG_ASSERT(*max_coulomb >= *min_coulomb);
  DEBUG_ASSERT(*max_exchange >= *min_exchange);
  
} // NodesUpperBound()

// For hybrid expansion, this may have to work on a query and list of refs
bool MultiTreeFock::CanPrune(MatrixTree* query, MatrixTree* reference, 
                             double* approx_coulomb, double* approx_exchange,
                             double* lost_error) {
  
  // to be implemented later
  
  if (!(query->row_shells()->single_momentum()) ||
      !(query->col_shells()->single_momentum()) ||
      !(reference->row_shells()->single_momentum()) ||
      !(reference->col_shells()->single_momentum())) {
    
    DEBUG_ONLY(*approx_coulomb = BIG_BAD_NUMBER);
    DEBUG_ONLY(*approx_exchange = BIG_BAD_NUMBER);
    
    return false;
    
  } // not all one momentum, don't prune
      
  double max_coulomb, min_coulomb;
  double max_exchange, min_exchange;
  NodeBounds(query, reference, &max_coulomb, &max_exchange, 
              &min_coulomb, &min_exchange);
  
  // create global bounds from the coulomb and exchange
  double max_fock = max_coulomb - min_exchange;
  double min_fock = min_coulomb - max_exchange;
  DEBUG_ASSERT(max_fock >= min_fock);
  
  
  // factor in the number of reference pairs to be accounted for 
  // this takes care of reference symmetry
  max_fock *= reference->num_pairs();
  min_fock *= reference->num_pairs();
  
  // how to bound the coulomb and exchange contributions together, while still
  // maintaining separate approximation values?
  
  *lost_error = 0.5 * (max_fock - min_fock);
  // need to account for the number of functions in a shell in this count
  double num_functions_per_shell = (double)eri::NumFunctions(reference->row_shells()->momenta().lo)
  * (double)eri::NumFunctions(reference->col_shells()->momenta().lo);
  double allowed_err = query->remaining_epsilon() * reference->num_pairs() 
                       * num_functions_per_shell
                       / query->remaining_references();
  if (*lost_error <= allowed_err) {
    
    //printf("lost_error: %g, allowed_err: %g\n", *lost_error, allowed_err);
    *approx_coulomb = 0.5 * (max_coulomb + min_coulomb);
    *approx_exchange = 0.5 * (max_exchange + min_exchange);
    num_approximations_++;
    
    return true;
    
  } // can prune
  else {
   
    DEBUG_ONLY(*approx_coulomb = BIG_BAD_NUMBER);
    DEBUG_ONLY(*approx_exchange = BIG_BAD_NUMBER);
    
    return false;
    
  } // can't prune
  
} // CanPrune()

void MultiTreeFock::DepthFirstRecursion(MatrixTree* query, 
                                        MatrixTree* reference) {
  
  // check for base case
  
  if (query->is_leaf() && reference->is_leaf()) {
    
    ComputeBaseCase(query, reference);
    
  } // base case
  else {

    double approx_coulomb;
    double approx_exchange;
    double lost_error;
    // attempt to prune
    if (CanPrune(query, reference, &approx_coulomb, 
                 &approx_exchange, &lost_error)) {
     
      // fill in approx vals 
      query->set_remaining_epsilon(query->remaining_epsilon() - lost_error);
      query->set_remaining_references(query->remaining_references() 
                                      - reference->num_pairs());
      query->add_coulomb_approx(approx_coulomb);
      query->add_exchange_approx(approx_exchange);
      
    } // pruning
    else if (SplitQuery(query, reference)) {
     
      // make split call
      success_t split;
      if (!(query->left())) {
        // the node hasn't been split
        split = matrix_tree_impl::SplitMatrixTree(query, shell_ptr_list_, 
                                                            density_matrix_);
        // bounds get passed down in splitting code
        
      }
      else {
        
        split = SUCCESS_PASS;
        
      }
      // don't forget to handle the internal bounds too
      
      if (split == SUCCESS_PASS) {
        
        PassBoundsDown_(query);
        
        DepthFirstRecursion(query->left(), reference);
        DepthFirstRecursion(query->right(), reference);
        
        PassBoundsUp_(query);
        
      }
      else if (reference->is_leaf()) {
        ComputeBaseCase(query, reference); 
      }
      else {
        // try to split the reference
        DEBUG_ASSERT(query->is_leaf());
        DepthFirstRecursion(query, reference);
      }
    } // split queries
    else {
      
      // split the reference
      success_t split;
      if (!(reference->left())) {
        // node hasn't been split
        split = matrix_tree_impl::SplitMatrixTree(reference, shell_ptr_list_,
                                                  density_matrix_);
        
      }
      else {
        split = SUCCESS_PASS;
      }
      //don't need to pass info down
      // TODO: need to pass info back up for this case?

      // may want to prioritize these somehow
      if (split == SUCCESS_PASS) {
        DepthFirstRecursion(query, reference->left());
        DepthFirstRecursion(query, reference->right());
      }
      else if (query->is_leaf()) {
        ComputeBaseCase(query, reference);
      }
      else {
        // try to split the query
        DEBUG_ASSERT(reference->is_leaf());
        DepthFirstRecursion(query, reference);
      }
    } // split references
    
  } // not base case
  
  
} // DepthFirstRecursion()




/*

void MultiTreeFock::ApplyPermutation(ArrayList<index_t>& old_from_new, 
                                     Matrix* mat) {

  DEBUG_ASSERT(old_from_new.size() == mat->n_cols());
  
  Matrix temp_mat;
  temp_mat.Init(mat->n_rows(), mat->n_cols());
  
  for (index_t i = 0; i < old_from_new.size(); i++) {
  
    Vector temp_vec;
    mat->MakeColumnVector(old_from_new[i], &temp_vec);
    ApplyPermutation(old_from_new, &temp_vec);
    temp_mat.CopyColumnFromMat(i, old_from_new[i], *mat);
  
  } // for i

  mat->CopyValues(temp_mat);
  
}

void MultiTreeFock::ApplyPermutation(ArrayList<index_t>& old_from_new, 
                                     Vector* vec) {

  DEBUG_ASSERT(old_from_new.size() == vec->length());
  
  Vector temp_vec;
  temp_vec.Init(vec->length());
  
  for (index_t i = 0; i < vec->length(); i++) {
    
    temp_vec[i] = (*vec)[old_from_new[i]];
    
  } // for i
  
  vec->CopyValues(temp_vec);
  
}

void MultiTreeFock::UnApplyPermutation(ArrayList<index_t>& old_from_new, 
                                       Matrix* mat) {

  DEBUG_ASSERT(old_from_new.size() == mat->n_cols());

  Matrix temp_mat;
  temp_mat.Init(mat->n_rows(), mat->n_cols());
  
  for (index_t i = 0; i < old_from_new.size(); i++) {
  
    Vector temp_vec;
    mat->MakeColumnVector(i, &temp_vec);
    UnApplyPermutation(old_from_new, &temp_vec);
    temp_mat.CopyColumnFromMat(old_from_new[i], i, *mat);
  
  } // for i

  mat->CopyValues(temp_mat);

}

void MultiTreeFock::UnApplyPermutation(ArrayList<index_t>& old_from_new, 
                                       Vector* vec) {

  DEBUG_ASSERT(old_from_new.size() == vec->length());

  Vector temp_vec;
  temp_vec.Init(vec->length());

  for (index_t i = 0; i < vec->length(); i++) {
  
    temp_vec[old_from_new[i]] = (*vec)[i];
  
  } // for i
  
  vec->CopyValues(temp_vec);

}
*/


///////////////////// public functions ////////////////////////////////////

void MultiTreeFock::Compute() {

  fx_timer_start(module_, "multi_time");

  printf("====Computing J and K====\n");
  
  //printf("remaining_references: %d\n", matrix_tree_->remaining_references());
  //printf("num_pairs: %d\n", matrix_tree_->num_pairs());
  DepthFirstRecursion(matrix_tree_, matrix_tree_);
  
  //tree_->Print();
  //matrix_tree_->Print();
  
  matrix_tree_impl::FormDenseMatrix(matrix_tree_, &coulomb_matrix_, 
                                    &exchange_matrix_);
  
  la::SubInit(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  
  fx_timer_stop(module_, "multi_time");
  
  fx_result_int(module_, "num_integrals_computed", num_integrals_computed_);
  fx_result_int(module_, "num_approximations", num_approximations_);

} // ComputeFockMatrix()

void MultiTreeFock::UpdateDensity(const Matrix& new_density) {
  
  density_matrix_.CopyValues(new_density);
  // this won't be correct when I switch to higher momentum
  //ApplyPermutation(old_from_new_centers_, &density_matrix_);
  
  //density_matrix_.PrintDebug();
  
  // Reset tree density bounds
  //ResetTree_(square_tree_);
  
  /*
  coulomb_matrix_.SetZero();
  exchange_matrix_.SetZero();
  fock_matrix_.SetZero();
  */
  coulomb_matrix_.Destruct();
  exchange_matrix_.Destruct();
  fock_matrix_.Destruct();
  
} // UpdateMatrices()


void MultiTreeFock::OutputFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                                     Matrix* exchange_out, 
                                     ArrayList<index_t>* old_from_new) {
  
  //printf("number_of_approximations_ = %d\n", number_of_approximations_);
  //printf("number_of_base_cases_ = %d\n\n", number_of_base_cases_);
  //fx_format_result(module_, "bandwidth", "%g", bandwidth_);
  
  
  if (fock_out) {
    fock_out->Copy(fock_matrix_);
    //UnApplyPermutation(old_from_new_centers_, fock_out);
  }
  if (coulomb_out) {
    coulomb_out->Copy(coulomb_matrix_);
    //UnApplyPermutation(old_from_new_centers_, coulomb_out);
  }
  if (exchange_out) {
    exchange_out->Copy(exchange_matrix_);
    //UnApplyPermutation(old_from_new_centers_, exchange_out);
  }
  
  if (old_from_new) {
    old_from_new->InitCopy(old_from_new_shells_);
  }
    
} // OutputFockMatrix()

void MultiTreeFock::OutputCoulomb(Matrix* coulomb_out) {
  
  if (coulomb_out) {
    coulomb_out->Copy(coulomb_matrix_);
    //UnApplyPermutation(old_from_new_centers_, coulomb_out);
  }
  
} // OutputCoulomb

void MultiTreeFock::OutputExchange(Matrix* exchange_out) {
  
  if (exchange_out) {
    exchange_out->Copy(exchange_matrix_);
    //UnApplyPermutation(old_from_new_centers_, exchange_out);
  }
  
} // OutputExchange




