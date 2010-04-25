#include "link.h"


bool Link::Prescreening_Sort_(BasisShell* shellA, BasisShell* shellB) {

  return ((shellA->max_schwartz_factor() * shellA->current_density_entry()) > 
          (shellB->max_schwartz_factor() * shellB->current_density_entry()));

}

bool Link::ShellPairSort_(ShellPair* shellA, ShellPair* shellB) {

  return(shellA->schwartz_factor() > shellB->schwartz_factor());

}


void Link::PrescreeningLoop_() {

  // Find "significant" bra and ket shell pairs
  // These need to be ordered in terms of their Schwartz estimates
  // I think shell_pair_list_ needs to be a list of lists
  // One list of shell pairs for each mu
  num_shell_pairs_ = eri::ComputeShellPairs(&shell_pair_list_, shell_list_, 
                                            shell_pair_cutoff_, &shell_max_, 
                                            &significant_sigma_for_nu_, 
                                            &num_significant_sigma_for_nu_, 
                                            density_matrix_);
  
  //shell_max_.PrintDebug("shell_max_");
  /*
  for (index_t i = 0; i < num_shells_; i++) {
    printf("num_significant_sigma_for_nu_[%d] = %d\n", i, 
           num_significant_sigma_for_nu_[i]);
  }
  */
  fx_result_int(module_, "num_shell_pairs", num_shell_pairs_);
  fx_result_int(module_, "num_shell_pairs_screened", 
                (num_shells_ * (num_shells_ - 1) / 2)
                 + num_shells_ - num_shell_pairs_);
  
  //// Find significant \nu for all \mu
  
  // I am supposed to loop over all mu in significant bra shell pairs and over 
  // all nu in significant ket shell pairs
  // Maybe I should just loop over the shells themselves and assume that each
  // shell appears in at least one significant shell pair
  for (index_t i = 0; i < num_shells_; i++) {
    
    current_mu_ = i;
    
    index_t next_ind = 0;
    
    ArrayList<index_t> significant_nu_index;
    significant_nu_index.Init(num_shells_);
    
    for (index_t j = 0; j < num_shells_; j++) {
      
      // I think this threshold should be the same as the others 
      // Not sure if this is the right density matrix entry
      // TODO: should this check all the density entries and take the max?
      /*
      double density_bound = -DBL_MAX;
      for (index_t a = 0; a < shell_list_[i].num_functions(); a++) {
        for (index_t b = 0; b < shell_list_[j].num_functions(); b++) {
          
          density_bound = max(density_bound, 
                              fabs(density.ref(shell_list_[i].matrix_index(a), 
                                               shell_list_[j].matrix_index(b))));
          
        } // for b
      } // for a
      
      DEBUG_ASSERT(density_bound >= 0.0);
       */
      
      double density_bound = eri::DensityBound(shell_list_[i], shell_list_[j], 
                                               density_matrix_);
      
      double density_check = density_bound * shell_max_[i] 
                              * shell_max_[j];
      if (density_check > threshold_) {
        
        // Store significant j for each i
        // What is the best way to do this?  This list will have to be sorted
        // This fails if there are more than num_shells_/2 significant elements
        significant_nu_index[next_ind] = j;
        
        // I need to store next_ind somehow in order to be able to iterate 
        // later
        next_ind++;
        
      }
      else {
        //printf("insignificant density: %d, %d, %g\n", i, j, density_check);
        num_insignificant_densities_++;
      }
      
    } // for j
    
    num_significant_nu_for_mu_[i] = next_ind;
    
    significant_nu_for_mu_[i] = (BasisShell**)malloc(next_ind * 
                                                    sizeof(BasisShell**)); 
    //BasisShell** significant_mu_pairs_[i] = significant_mu_pairs_[i];
    
    for (index_t k = 0; k < next_ind; k++) {
      
      significant_nu_for_mu_[i][k] = shell_list_.begin() + 
          significant_nu_index[k];
      significant_nu_for_mu_[i][k]->set_max_schwartz_factor(shell_max_[k]);
      
      //significant_nu_for_mu_[i][k]->set_current_density_entry(density_matrix_.ref(i,k));
      double density_entry = eri::DensityBound(shell_list_[i], 
                                               *(shell_list_.begin() + significant_nu_index[k]),
                                               density_matrix_);
      significant_nu_for_mu_[i][k]->set_current_density_entry(density_entry);
      
    } // for k
    
    // sort significant_nu_for_mu
    
    std::sort(significant_nu_for_mu_[i], significant_nu_for_mu_[i]+next_ind, 
              Link::Prescreening_Sort_);
              
    
    std::sort(significant_sigma_for_nu_[i], 
              significant_sigma_for_nu_[i]+num_significant_sigma_for_nu_[i], 
              Link::ShellPairSort_);

  } // for i
  
}


void Link::Compute() {

  fx_timer_start(module_, "LinK_time");

  // Prescreening loop
  printf("====LinK Prescreening====\n");
  fx_timer_start(module_, "prescreening");
  PrescreeningLoop_();
  fx_timer_stop(module_, "prescreening");

  
  printf("====LinK Screening and Computing Integrals====\n");
  fx_timer_start(module_, "integrals");
  // loop over bra shell pairs
  
  for (index_t shell_pair_ind = 0; shell_pair_ind < num_shell_pairs_; 
       shell_pair_ind++) {
       
    ShellPair& mu_lambda = shell_pair_list_[shell_pair_ind];
    
    // these are the indices into the shell list
    // they're correct, just need to make sure they're not used to access
    // the density matrix
    index_t mu_ind = mu_lambda.M_index();
    index_t lambda_ind = mu_lambda.N_index();
    
    //printf("mu_ind: %d, lambda_ind: %d\n", mu_ind, lambda_ind);
    
    index_t num_mu_integrals = 0;
    index_t num_lambda_integrals = 0;
    ArrayList<index_t> mu_integrals;
    ArrayList<index_t> lambda_integrals;
    
    mu_integrals.Init();
    lambda_integrals.Init();
  
    // loop over nu corresponding to mu
    // what is num_nu? 
    index_t num_nu = num_significant_nu_for_mu_[mu_ind];
    //printf("num_nu: %d\n", num_nu);
    for (index_t sorted_nu_ind = 0; sorted_nu_ind < num_nu; sorted_nu_ind++) {
    
      index_t significant_sigmas = 0;
      
      // need to get this from the first sorted list
      BasisShell* nu_shell = significant_nu_for_mu_[mu_ind][sorted_nu_ind];
      
      // this doesn't exist anymore - what was it for?
      index_t nu_ind = nu_shell->list_index();
    
      index_t num_sigma = num_significant_sigma_for_nu_[nu_ind];

      //printf("num_sigma: %d\n", num_sigma);
      
      for (index_t sorted_sigma_ind = 0; sorted_sigma_ind < num_sigma; 
           sorted_sigma_ind++) {
      
        ShellPair* nu_sigma = significant_sigma_for_nu_[nu_ind][sorted_sigma_ind];
        
        // need to replace this with appropriate integral call
        // does this need to take into account the other density entries?
        //double density_bound = fabs(density_matrix_.ref(mu_ind, nu_ind));
        double density_bound = eri::DensityBound(*(mu_lambda.M_Shell()), *nu_shell, 
                                                 density_matrix_);
        double bound = density_bound * mu_lambda.schwartz_factor() * 
            nu_sigma->schwartz_factor();
	//printf("bound: %g\n", bound);
        if (bound > threshold_) {
      
          // store or compute the eri
          // need to change this to storing it so I can merge the lists later
          // create a list of index_t, list[i] = j, where shell_pairs_[j] has
          // significant integrals relative to mu_lambda
          // this list will need to be sorted, then I can use std::set_union
          
          // how am I going to get the index of nu_sigma in the list of shell 
          // pairs?  
          mu_integrals.PushBack();
          mu_integrals[num_mu_integrals] = nu_sigma->list_index();
          num_mu_integrals++;
          
          significant_sigmas++;
          
        }
        // leave sigma loop, since it is sorted
        else {
          //printf("left sigma loop, bound: %g\n", bound);
          num_neglected_sigma_ += num_sigma - sorted_sigma_ind - 1;
          break;
        }
        
      } // for sigma_ind
      
      // exit loop if this nu has no sigmas that count
      if (significant_sigmas == 0) {
        //printf("left nu loop\n");
        num_neglected_nu_ += num_nu - sorted_nu_ind - 1;
        break;
      }
    
    } // for nu_ind

    // loop over nu corresponding to lambda and do same
    // why is this necessary as well?  
    // I think it's important for the symmetry, lambda mu never shows up as a 
    // shell pair
    // shouldn't do this if lambda == mu

    // currently assumes that there is at least one significant integral for 
    // for this shell pair - this needs to be fixed
    
    if (mu_ind != lambda_ind) {
      //printf("lambda loop\n");
      num_nu = num_significant_nu_for_mu_[lambda_ind];
      
      for (index_t sorted_nu_ind = 0; sorted_nu_ind < num_nu; sorted_nu_ind++) {
        
        index_t significant_sigmas = 0;
        
        // need to get this from the first sorted list
        // should this be lambda?
        // changed this to lambda, I think this is right
        // seems to be okay, tested on the first three equilibrated helium sets
        BasisShell* nu_shell = significant_nu_for_mu_[lambda_ind][sorted_nu_ind];
        
        index_t nu_ind = nu_shell->list_index();
        
        index_t num_sigma = num_significant_sigma_for_nu_[nu_ind];
        
        for (index_t sorted_sigma_ind = 0; sorted_sigma_ind < num_sigma; 
             sorted_sigma_ind++) {
          
          ShellPair* nu_sigma = significant_sigma_for_nu_[nu_ind][sorted_sigma_ind];
          
          //index_t sigma_ind = nu_sigma->N_index();
          
          // shouldn't this involve lambda?
          //double density_bound = fabs(density_matrix_.ref(mu_ind, nu_ind));
          // IMPORTANT: changed this to use lambda's density matrix entry
          double density_bound = eri::DensityBound(*(mu_lambda.N_Shell()), *nu_shell, 
                                                   density_matrix_);
          if (density_bound * mu_lambda.schwartz_factor() * 
              nu_sigma->schwartz_factor() > threshold_) {
            
            lambda_integrals.PushBack();
            //printf("list_index: %d\n", nu_sigma->list_index());
            lambda_integrals[num_lambda_integrals] = nu_sigma->list_index();
            num_lambda_integrals++;
                        
            significant_sigmas++;
            
          }
          // leave sigma loop, since it is sorted
          else {
            //printf("left sigma loop\n");
            num_neglected_sigma_ += num_sigma - sorted_sigma_ind - 1;
            break;
          }
          
        } // for sigma_ind
        
        // exit loop if this nu has no sigmas that count
        if (significant_sigmas == 0) {
          //printf("left nu loop\n");
          num_neglected_nu_ += num_nu - sorted_nu_ind - 1;
          break;
        }
        
      } // for nu_ind
    
    } // if lambda != mu
    else {
      
      // make the list non-void for the union step below
      lambda_integrals.PushBack();
      lambda_integrals[num_lambda_integrals] = mu_integrals[0];
      num_lambda_integrals++;
      
    }
    

    // compute significant integrals
    // may just do this inside loop

    // sort before calling union
    std::sort(mu_integrals.begin(), mu_integrals.begin()+num_mu_integrals);
    std::sort(lambda_integrals.begin(), 
              lambda_integrals.begin() + num_lambda_integrals);

    index_t* end_integrals;
    ArrayList<index_t> integral_list;
    integral_list.Init(num_shell_pairs_);
    
    end_integrals = std::set_union(mu_integrals.begin(), 
                                   mu_integrals.begin()+num_mu_integrals, 
                                   lambda_integrals.begin(), 
                                   lambda_integrals.begin() + num_lambda_integrals, 
                                   integral_list.begin());
                                   
    index_t num_integrals = end_integrals - integral_list.begin();
    //printf("num_integrals = %d\n", num_integrals);
    
    // compute the integrals
    
    for (index_t int_ind = 0; int_ind < num_integrals; int_ind++) {
    
      ShellPair nu_sigma = shell_pair_list_[integral_list[int_ind]];
      
      IntegralTensor integrals;
      eri::ComputeShellIntegrals(mu_lambda, nu_sigma, &integrals);
      
      num_integrals_computed_++;
      
      // contract with mu, nu; mu, sigma; lambda, nu; lambda, sigma
      // sum into lambda, sigma; lambda, nu; mu, sigma; mu, nu
      Matrix exchange_ik;
      exchange_ik.Init(mu_lambda.M_Shell()->num_functions(), 
                       nu_sigma.M_Shell()->num_functions());
      exchange_ik.SetZero();
      
      Matrix* exchange_jk;
      Matrix* exchange_il;
      Matrix* exchange_jl;
      
      // not sure if this will work with the references
      // if i != j
      if (mu_lambda.M_Shell() != mu_lambda.N_Shell()) {
        exchange_jk = new Matrix();
        exchange_jk->Init(mu_lambda.N_Shell()->num_functions(), 
                          nu_sigma.M_Shell()->num_functions());
        exchange_jk->SetZero();
      }
      else {
        exchange_jk = NULL;
      }
      
      // if k != l
      if (nu_sigma.N_Shell() != nu_sigma.M_Shell()) {
        exchange_il = new Matrix();
        exchange_il->Init(mu_lambda.M_Shell()->num_functions(), 
                          nu_sigma.N_Shell()->num_functions());
        exchange_il->SetZero();
        
      }
      else {
        exchange_il = NULL;
      }
      
      if (mu_lambda.M_Shell() != mu_lambda.N_Shell() && 
          nu_sigma.N_Shell() != nu_sigma.M_Shell()) {
        exchange_jl = new Matrix();
        exchange_jl->Init(mu_lambda.N_Shell()->num_functions(), 
                          nu_sigma.N_Shell()->num_functions());
        exchange_jl->SetZero();
        
      }
      else {
        exchange_jl = NULL; 
      }
      
      integrals.ContractExchange(mu_lambda.M_Shell()->matrix_indices(),
                                 mu_lambda.N_Shell()->matrix_indices(),
                                 nu_sigma.M_Shell()->matrix_indices(),
                                 nu_sigma.N_Shell()->matrix_indices(),
                                 density_matrix_, &exchange_ik, exchange_jk, 
                                 exchange_il, exchange_jl);
      
      eri::AddSubmatrix(mu_lambda.M_Shell()->matrix_indices(), 
                        nu_sigma.M_Shell()->matrix_indices(),
                        exchange_ik, &exchange_matrix_);
      
      if (exchange_jk) {
        eri::AddSubmatrix(mu_lambda.N_Shell()->matrix_indices(), 
                          nu_sigma.M_Shell()->matrix_indices(),
                          *exchange_jk, &exchange_matrix_);
	delete exchange_jk;
      }
      
      if (exchange_il) {
        eri::AddSubmatrix(mu_lambda.M_Shell()->matrix_indices(), 
                          nu_sigma.N_Shell()->matrix_indices(),
                          *exchange_il, &exchange_matrix_);
	delete exchange_il;
      }
      
      if (exchange_jl) {
        eri::AddSubmatrix(mu_lambda.N_Shell()->matrix_indices(), 
                          nu_sigma.N_Shell()->matrix_indices(),
                          *exchange_jl, &exchange_matrix_);
	delete exchange_jl;
      }
      
      
      //// old (only s function) code
      /*
      double lambda_sigma_int = density_matrix_.ref(lambda_ind, sigma_ind) * integral;
      
      double mu_nu_exc = exchange_matrix_.ref(mu_ind, nu_ind);
      
      exchange_matrix_.set(mu_ind, nu_ind, mu_nu_exc + lambda_sigma_int);
      if (sigma_ind != nu_ind) {
      
        double mu_sigma_exc = exchange_matrix_.ref(mu_ind, sigma_ind);
        double lambda_nu_int = density_matrix_.ref(lambda_ind, nu_ind) * integral;
        
        exchange_matrix_.set(mu_ind, sigma_ind, 
                             mu_sigma_exc + lambda_nu_int);
      }
      if (lambda_ind != mu_ind){

        double mu_sigma_int = density_matrix_.ref(mu_ind, sigma_ind) * integral;
        double lambda_nu_exc = exchange_matrix_.ref(lambda_ind, nu_ind);

        exchange_matrix_.set(lambda_ind, nu_ind, 
                             lambda_nu_exc + mu_sigma_int);
      }
      if ((lambda_ind != mu_ind) && (nu_ind != sigma_ind)) {
      
        double mu_nu_int = density_matrix_.ref(mu_ind, nu_ind) * integral;
        double lambda_sigma_exc = exchange_matrix_.ref(lambda_ind, sigma_ind);
        
        exchange_matrix_.set(lambda_ind, sigma_ind, 
                             lambda_sigma_exc + mu_nu_int);
      }
      */ 
       
      
    
    } // for int_ind

  } // for bra shell pairs
  
  fx_timer_stop(module_, "integrals");

  fx_timer_stop(module_, "LinK_time");

  first_computation = false;
  
} // ComputeExchangeMatrix()

void Link::UpdateDensity(const Matrix& new_density) {
  
  // not sure if density will need to be freed first
  density_matrix_.CopyValues(new_density);
  
  if (!first_computation) {
    for (index_t i = 0; i < num_shells_; i++) {
      /*for (index_t j = 0; j < num_significant_nu_for_mu_[i]; j++) {
        free(significant_nu_for_mu_[i][j]);
      }*/
      free(significant_nu_for_mu_[i]);
    }
    free(significant_nu_for_mu_);
    
    for (index_t i = 0; i < num_shells_; i++) {
      /*for (index_t j = 0; j < num_significant_sigma_for_nu_[i]; j++) {
        free(significant_sigma_for_nu_[i][j]);
      }*/
      free(significant_sigma_for_nu_[i]);
    }
    free(significant_sigma_for_nu_);
    
    significant_nu_for_mu_ = 
    (BasisShell***)malloc(num_shells_*sizeof(BasisShell**));
    
    significant_sigma_for_nu_ = 
    (ShellPair***)malloc(num_shells_*sizeof(ShellPair**));
    
    num_significant_nu_for_mu_.Clear();
    num_significant_nu_for_mu_.Init(num_shells_);
    
    num_significant_sigma_for_nu_.Clear();
    num_significant_sigma_for_nu_.Init(num_shells_);
    
    // does this free the shell pairs in it?
    // doesn't look like it matters
    shell_pair_list_.Clear();
  
    shell_max_.Destruct();
    
  }
    
  exchange_matrix_.SetZero();
  
  
  /*
  num_neglected_sigma_ = 0;
  num_neglected_nu_ = 0;
  */
} // UpdateDensity

void Link::OutputExchange(Matrix* exc_out) {
  
  fx_result_int(module_, "num_neglected_sigma", num_neglected_sigma_);
  fx_result_int(module_, "num_neglected_nu", num_neglected_nu_);
  fx_result_int(module_, "num_integrals_computed", num_integrals_computed_);
  fx_result_int(module_, "num_insignificant_densities", 
                num_insignificant_densities_);
  
  la::Scale(0.5, &exchange_matrix_);
  exc_out->Copy(exchange_matrix_);
  
} // OutputFockMatrix

