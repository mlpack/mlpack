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
  
  query->left()->add_approx(query->approx_val());
  query->right()->add_approx(query->approx_val());
  // since it's been passed down now
  query->set_approx_val(0.0);
  
}



void MultiTreeFock::ComputeBaseCaseCoulomb(MatrixTree* query, 
                                           MatrixTree* reference) {
  
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
                        coulomb_ij, query->entries());
      
      // the "below diagonal" (ji) entry should be taken care of in the
      // matrix reconstruction code
      
      internal_col_index += j_shell->num_functions();
      
    } // for j
    
    internal_row_index += i_shell->num_functions();
      
  } // for i
  
  num_coulomb_base_cases_++;
  query->set_remaining_references(query->remaining_references() 
                                  - reference->num_pairs());
  
} // ComputeBaseCaseCoulomb()

void MultiTreeFock::ComputeBaseCaseExchange(MatrixTree* query, 
                                            MatrixTree* reference) {
  
  index_t internal_row_index = 0;
  
  for (index_t i = query->row_shells()->begin(); i < query->row_shells()->end();
       i++) {
    
    BasisShell* i_shell = shell_ptr_list_[i];
    
    index_t internal_col_index = 0;
    
    for (index_t j = query->col_shells()->begin(); j < query->col_shells()->end(); 
         j++) {
      
      BasisShell* j_shell = shell_ptr_list_[j];
      
      for (index_t k = reference->row_shells()->begin(); 
           k < reference->row_shells()->end(); k++) {
        
        BasisShell* k_shell = shell_ptr_list_[k];
        
        for (index_t l = reference->col_shells()->begin(); 
             l < reference->col_shells()->end(); l++) {
          
          BasisShell* l_shell = shell_ptr_list_[l];
          
          IntegralTensor exchange_ints;
          
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
                            exchange_ik, query->entries());
          
        } // for l
        
      } // for k
      
      internal_col_index += j_shell->num_functions();
      
    } // for j
    
    internal_row_index += i_shell->num_functions();
    
  } // for i
  
  query->set_remaining_references(query->remaining_references() 
                                  - reference->num_pairs());
  
  num_exchange_base_cases_++;
  
} // ComputeBaseCaseExchange()

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

void MultiTreeFock::NodeBoundsCoulomb(MatrixTree* query, MatrixTree* reference,
                                      double* max_coulomb, double* min_coulomb) {
  
  
  // compute these in ERI somehow
  double max_coulomb_integral, min_coulomb_integral;
  
  // make calls to eri
  // or possibly create an eri_bounds namespace to keep things neat
  // choose which function to call based on momenta of the shells
  max_coulomb_integral = eri_bounds::BoundIntegrals(query->row_shells(),
                                                    query->col_shells(),
                                                    reference->row_shells(),
                                                    reference->col_shells());
  
  // I believe this is taken care of by multiplying by reference->num_pairs()
  // later
  /*
  if (!(reference->on_diagonal())) {
    // account for the reference symmetry
    max_coulomb_integral *= 2.0;
  }
   */
  
  // need to come up with something better than this
  int total_momentum = query->row_shells()->momenta().lo
                       + query->col_shells()->momenta().lo
                       + reference->row_shells()->momenta().lo
                       + reference->col_shells()->momenta().lo;
  if (total_momentum > 0) {
    min_coulomb_integral = -1 * max_coulomb_integral;
  }
  else {
    // is this correct?  yes, the integral is strictly positive
    min_coulomb_integral = 0.0;
  }
  
  // factor in the number of functions
  double num_functions_per_shell = (double)eri::NumFunctions(reference->row_shells()->momenta().lo)
  * (double)eri::NumFunctions(reference->col_shells()->momenta().lo);
  max_coulomb_integral *= num_functions_per_shell;
  min_coulomb_integral *= num_functions_per_shell;
  
  // factor in the density
  if (reference->density_bounds().hi >= 0.0) {
    
    *max_coulomb = reference->density_bounds().hi * max_coulomb_integral;
    
  }
  else {
    
    *max_coulomb = reference->density_bounds().hi * min_coulomb_integral;
    
  }
  
  if (reference->density_bounds().lo >= 0.0) {
    
    *min_coulomb = reference->density_bounds().lo * min_coulomb_integral;
    
  }
  else {

    *min_coulomb = reference->density_bounds().lo * max_coulomb_integral;
    
  }
  
  DEBUG_ASSERT(*max_coulomb >= *min_coulomb);
  
} // NodeBoundsCoulomb()

void MultiTreeFock::NodeBoundsExchange(MatrixTree* query, MatrixTree* reference,
                                       double* max_exchange, 
                                       double* min_exchange) {
  
  
  double max_exchange_integral, min_exchange_integral;
  
  // TODO: is this right?  What about the reference symmetry?
  // Do I need to compute both sides and take the max?
  // Only for off diagonal queries?
  max_exchange_integral = eri_bounds::BoundIntegrals(query->row_shells(),
                                                     reference->row_shells(),
                                                     query->col_shells(),
                                                     reference->col_shells());
  
  // TODO: how to deal with this when multiplying by reference->num_pairs() ?
  // if its not on the diagonal, there are an equal number above and below it
  // therefore, the two estimates should be weighted equally 
  // thus, I should multiply each by 0.5
  if (!(reference->on_diagonal())) {
    
    max_exchange_integral += eri_bounds::BoundIntegrals(query->row_shells(),
                                                        reference->col_shells(),
                                                        query->col_shells(),
                                                        reference->row_shells());
    
    // deals with reference symmetry when multiplying by reference->num_pairs()
    max_exchange_integral *= 0.5;
    
  }
  // What about the partially on diagonal case (i.e. rectangular node)?
  // the on diagonal integrals should be greater than any of the off diagonal 
  // ones (i.e. they have the self-overlap)
  // If this is true, then I'm effectively using the on-diagonal integral
  // as an upper bound for both off-diagonal integrals, which means it should
  // be counted twice in the multiplication by num_pairs()
  
  
  int total_momentum = query->row_shells()->momenta().lo
                       + query->col_shells()->momenta().lo
                       + reference->row_shells()->momenta().lo
                       + reference->col_shells()->momenta().lo;
  if (total_momentum > 0) {
    min_exchange_integral = -1 * max_exchange_integral;
  }
  else {
    // is this correct?  yes, the integral is strictly positive
    min_exchange_integral = 0.0;
  }
  
  // factor in the number of functions
  double num_functions_per_shell = (double)eri::NumFunctions(reference->row_shells()->momenta().lo)
  * (double)eri::NumFunctions(reference->col_shells()->momenta().lo);
  max_exchange_integral *= num_functions_per_shell;
  min_exchange_integral *= num_functions_per_shell;
  
  // factor in the density
  if (reference->density_bounds().hi >= 0.0) {
    
    *max_exchange = reference->density_bounds().hi * max_exchange_integral;
    
  }
  else {
    
    *max_exchange = reference->density_bounds().hi * min_exchange_integral;
    
  }
  
  if (reference->density_bounds().lo >= 0.0) {
    
    *min_exchange = reference->density_bounds().lo * min_exchange_integral;
    
  }
  else {
    
    *min_exchange = reference->density_bounds().lo * max_exchange_integral;
    
  }
  
  DEBUG_ASSERT(*max_exchange >= *min_exchange);
  
} // NodeBoundsExchange()


// For hybrid expansion, this may have to work on a query and list of refs
bool MultiTreeFock::CanPruneCoulomb(MatrixTree* query, MatrixTree* reference, 
                                    double* approx_val, double* lost_error) {
  
  if (!(query->row_shells()->single_momentum()) ||
      !(query->col_shells()->single_momentum()) ||
      !(reference->row_shells()->single_momentum()) ||
      !(reference->col_shells()->single_momentum())) {
    
    DEBUG_ONLY(*approx_val = BIG_BAD_NUMBER);
    
    return false;
    
  } // not all one momentum, don't prune
  
  if (prescreening_cutoff_ > 0.0) {
    
    double row_exp = query->row_shells()->momenta().lo;
    double col_exp = query->col_shells()->momenta().lo;
    double AB_dist_sq = query->row_shells()->bound().MinDistanceSq(query->col_shells()->bound());
    double query_overlap = eri::ComputeShellOverlap(AB_dist_sq, row_exp, col_exp);
    
    row_exp = reference->row_shells()->momenta().lo;
    col_exp = reference->col_shells()->momenta().lo;
    AB_dist_sq = reference->row_shells()->bound().MinDistanceSq(reference->col_shells()->bound());
    double reference_overlap = eri::ComputeShellOverlap(AB_dist_sq, row_exp, col_exp);
    
    if (query_overlap < prescreening_cutoff_ 
        || reference_overlap < prescreening_cutoff_) {
     
      *approx_val = 0.0;
      *lost_error = 0.0;
      
      num_coulomb_prescreening_prunes_++;
      
      return true;
      
    }
    
  }
      
  double max_coulomb, min_coulomb;
  NodeBoundsCoulomb(query, reference, &max_coulomb, &min_coulomb);
  DEBUG_ASSERT(max_coulomb >= min_coulomb);
  
  // factor in the number of reference pairs to be accounted for 
  // this takes care of reference symmetry
  max_coulomb *= reference->num_pairs();
  min_coulomb *= reference->num_pairs();
  
  // how to bound the coulomb and exchange contributions together, while still
  // maintaining separate approximation values?
  
  *lost_error = 0.5 * (max_coulomb - min_coulomb);
  
  //double num_functions_per_shell = (double)eri::NumFunctions(reference->row_shells()->momenta().lo)
  //* (double)eri::NumFunctions(reference->col_shells()->momenta().lo);
  
  // can't multiply by num_functions per shell here
  // epsilon is divided per shell
  double allowed_err = query->remaining_epsilon() * reference->num_pairs() 
                       / query->remaining_references();
  
  if (*lost_error <= allowed_err) {
    
    //printf("lost_error: %g, allowed_err: %g\n", *lost_error, allowed_err);
    *approx_val = 0.5 * (max_coulomb + min_coulomb);
    num_coulomb_approximations_++;
    
    return true;
    
  } // can prune
  else {
   
    DEBUG_ONLY(*approx_val = BIG_BAD_NUMBER);
    
    return false;
    
  } // can't prune
  
} // CanPruneCoulomb()

// For hybrid expansion, this may have to work on a query and list of refs
bool MultiTreeFock::CanPruneExchange(MatrixTree* query, MatrixTree* reference, 
                                     double* approx_exchange,
                                     double* lost_error) {
  
  // to be implemented later
  
  if (!(query->row_shells()->single_momentum()) ||
      !(query->col_shells()->single_momentum()) ||
      !(reference->row_shells()->single_momentum()) ||
      !(reference->col_shells()->single_momentum())) {
    
    DEBUG_ONLY(*approx_exchange = BIG_BAD_NUMBER);
    
    return false;
    
  } // not all one momentum, don't prune
  
  if (prescreening_cutoff_ > 0.0) {
  
    double row_exp = query->row_shells()->momenta().lo;
    double col_exp = reference->row_shells()->momenta().lo;
    double AB_dist_sq = query->row_shells()->bound().MinDistanceSq(reference->row_shells()->bound());
    double bra_overlap = eri::ComputeShellOverlap(AB_dist_sq, row_exp, col_exp);
    
    row_exp = query->col_shells()->momenta().lo;
    col_exp = reference->col_shells()->momenta().lo;
    AB_dist_sq = query->col_shells()->bound().MinDistanceSq(reference->col_shells()->bound());
    double ket_overlap = eri::ComputeShellOverlap(AB_dist_sq, row_exp, col_exp);
    
    // if the reference is off the diagonal, then these could be significant
    if (!(reference->on_diagonal())) {
     
      row_exp = query->row_shells()->momenta().lo;
      col_exp = reference->col_shells()->momenta().lo;
      AB_dist_sq = query->row_shells()->bound().MinDistanceSq(reference->col_shells()->bound());
      double sym_bra_overlap = eri::ComputeShellOverlap(AB_dist_sq, row_exp, col_exp);
      bra_overlap = max(bra_overlap, sym_bra_overlap);
      
      row_exp = query->col_shells()->momenta().lo;
      col_exp = reference->row_shells()->momenta().lo;
      AB_dist_sq = query->col_shells()->bound().MinDistanceSq(reference->row_shells()->bound());
      double sym_ket_overlap = eri::ComputeShellOverlap(AB_dist_sq, row_exp, col_exp);
      ket_overlap = max(ket_overlap, sym_ket_overlap);
      
    }
    
    if (bra_overlap < prescreening_cutoff_ 
        || ket_overlap < prescreening_cutoff_) {
      
      *approx_exchange = 0.0;
      *lost_error = 0.0;
      
      num_exchange_prescreening_prunes_++;
      
      return true;
      
    }
    
  }
  
  double max_exchange, min_exchange;
  NodeBoundsExchange(query, reference, &max_exchange, &min_exchange);
  
  // Exchange has a factor of 1/2
  max_exchange *= 0.5;
  min_exchange *= 0.5;
  
  DEBUG_ASSERT(max_exchange >= min_exchange);
  
  
  // factor in the number of reference pairs to be accounted for 
  // this takes care of reference symmetry
  max_exchange *= reference->num_pairs();
  min_exchange *= reference->num_pairs();
  
  // how to bound the coulomb and exchange contributions together, while still
  // maintaining separate approximation values?
  
  *lost_error = 0.5 * (max_exchange - min_exchange);
  // need to account for the number of functions in a shell in this count
  //double num_functions_per_shell = (double)eri::NumFunctions(reference->row_shells()->momenta().lo)
  //* (double)eri::NumFunctions(reference->col_shells()->momenta().lo);
  double allowed_err = query->remaining_epsilon() * reference->num_pairs() 
                       / query->remaining_references();
  if (*lost_error <= allowed_err) {
    
    //printf("lost_error: %g, allowed_err: %g\n", *lost_error, allowed_err);
    *approx_exchange = 0.5 * (max_exchange + min_exchange);
    num_exchange_approximations_++;
    
    return true;
    
  } // can prune
  else {
    
    DEBUG_ONLY(*approx_exchange = BIG_BAD_NUMBER);
    
    return false;
    
  } // can't prune
  
} // CanPruneExchange()

void MultiTreeFock::DepthFirstRecursionCoulomb(MatrixTree* query, 
                                               MatrixTree* reference) {
  
  // check for base case
  
  if (query->is_leaf() && reference->is_leaf()) {
    
    ComputeBaseCaseCoulomb(query, reference);
    
  } // base case
  else {

    double approx_val;
    double lost_error;
    // attempt to prune
    if (CanPruneCoulomb(query, reference, &approx_val, &lost_error)) {
     
      // fill in approx vals 
      query->set_remaining_epsilon(query->remaining_epsilon() - lost_error);
      query->set_remaining_references(query->remaining_references() 
                                      - reference->num_pairs());
      query->add_approx(approx_val);
      
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
        
        DepthFirstRecursionCoulomb(query->left(), reference);
        DepthFirstRecursionCoulomb(query->right(), reference);
        
        PassBoundsUp_(query);
        
      }
      else if (reference->is_leaf()) {
        ComputeBaseCaseCoulomb(query, reference); 
      }
      else {
        // try to split the reference
        DEBUG_ASSERT(query->is_leaf());
        DepthFirstRecursionCoulomb(query, reference);
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
        DepthFirstRecursionCoulomb(query, reference->left());
        DepthFirstRecursionCoulomb(query, reference->right());
      }
      else if (query->is_leaf()) {
        ComputeBaseCaseCoulomb(query, reference);
      }
      else {
        // try to split the query
        DEBUG_ASSERT(reference->is_leaf());
        DepthFirstRecursionCoulomb(query, reference);
      }
    } // split references
    
  } // not base case
  
  
} // DepthFirstRecursionCoulomb()


void MultiTreeFock::DepthFirstRecursionExchange(MatrixTree* query, 
                                                MatrixTree* reference) {
  
  // check for base case
  
  if (query->is_leaf() && reference->is_leaf()) {
    
    ComputeBaseCaseExchange(query, reference);
    
  } // base case
  else {
    
    double approx_val;
    double lost_error;
    // attempt to prune
    if (CanPruneExchange(query, reference, &approx_val, &lost_error)) {
      
      // fill in approx vals 
      query->set_remaining_epsilon(query->remaining_epsilon() - lost_error);
      query->set_remaining_references(query->remaining_references() 
                                      - reference->num_pairs());
      query->add_approx(approx_val);
      
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
        
        DepthFirstRecursionExchange(query->left(), reference);
        DepthFirstRecursionExchange(query->right(), reference);
        
        PassBoundsUp_(query);
        
      }
      else if (reference->is_leaf()) {
        ComputeBaseCaseExchange(query, reference); 
      }
      else {
        // try to split the reference
        DEBUG_ASSERT(query->is_leaf());
        DepthFirstRecursionExchange(query, reference);
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
        DepthFirstRecursionExchange(query, reference->left());
        DepthFirstRecursionExchange(query, reference->right());
      }
      else if (query->is_leaf()) {
        ComputeBaseCaseExchange(query, reference);
      }
      else {
        // try to split the query
        DEBUG_ASSERT(reference->is_leaf());
        DepthFirstRecursionExchange(query, reference);
      }
    } // split references
    
  } // not base case
  
} // DepthFirstRecursionExchange()


///////////////////// public functions ////////////////////////////////////

void MultiTreeFock::Compute() {

  fx_timer_start(module_, "multi_time");

  fx_timer_start(module_, "coulomb_time");
  printf("====Computing J ====\n");
  
  //printf("remaining_references: %d\n", matrix_tree_->remaining_references());
  //printf("num_pairs: %d\n", matrix_tree_->num_pairs());
  DepthFirstRecursionCoulomb(matrix_tree_, matrix_tree_);
  
  //tree_->Print();
  //matrix_tree_->Print();
  
  matrix_tree_impl::FormDenseMatrix(matrix_tree_, &coulomb_matrix_);
  fx_timer_stop(module_, "coulomb_time");
  
  printf("====Computing K ====\n");

  fx_timer_start(module_, "exchange_time");
  
  delete matrix_tree_;
  matrix_tree_ = matrix_tree_impl::CreateMatrixTree(tree_, shell_ptr_list_, 
                                                    density_matrix_);
  
  matrix_tree_->set_remaining_references(matrix_tree_->num_pairs());
  matrix_tree_->set_remaining_epsilon(epsilon_ * 0.5);
  
  DepthFirstRecursionExchange(matrix_tree_, matrix_tree_);
  
  matrix_tree_impl::FormDenseMatrix(matrix_tree_, &exchange_matrix_);
  
  fx_timer_stop(module_, "exchange_time");
  
  // exchange should already have the factor of 1/2
  la::SubInit(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  
  fx_timer_stop(module_, "multi_time");
  
  fx_result_int(module_, "num_integrals_computed", num_integrals_computed_);
  fx_result_int(module_, "num_coulomb_approximations", 
                num_coulomb_approximations_);
  fx_result_int(module_, "num_exchange_approximations", 
                num_exchange_approximations_);
  fx_result_int(module_, "num_coulomb_base_cases", 
                num_coulomb_base_cases_);
  fx_result_int(module_, "num_exchange_base_cases", 
                num_exchange_base_cases_);
  fx_result_int(module_, "num_coulomb_prescreening_prunes", 
                num_coulomb_prescreening_prunes_);
  fx_result_int(module_, "num_exchange_prescreening_prunes", 
                num_exchange_prescreening_prunes_);
  
  delete matrix_tree_;

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




