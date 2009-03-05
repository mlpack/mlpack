#include "link.h"


bool Link::Prescreening_Sort_(BasisShell* shellA, BasisShell* shellB) {

  return ((shellA->max_schwartz_factor() * shellA->current_density_entry()) < 
          (shellB->max_schwartz_factor() * shellB->current_density_entry()));

}

bool Link::ShellPairSort_(ShellPair* shellA, ShellPair* shellB) {

  return(shellA->schwartz_factor() < shellB->schwartz_factor());

}


void Link::PrescreeningLoop_() {

  // Find "significant" bra and ket shell pairs
  // These need to be ordered in terms of their Schwartz estimates
  // I think shell_pair_list_ needs to be a list of lists
  // One list of shell pairs for each mu
  num_shell_pairs_ = eri::ComputeShellPairs(&shell_pair_list_, shell_list_, 
                                            shell_pair_cutoff_, &shell_max_, 
                                            significant_sigma_for_nu_, 
                                            &num_significant_sigma_for_nu_);
                                            
  
  
  
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
      if (density_matrix_.ref(i,j) * shell_max_[i] * shell_max_[j] > threshold_) {
        
        // Store significant j for each i
        // What is the best way to do this?  This list will have to be sorted
        // This fails if there are more than num_shells_/2 significant elements
        significant_nu_index[next_ind] = j;
        
        // I need to store next_ind somehow in order to be able to iterate 
        // later
        next_ind++;
        
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
      significant_nu_for_mu_[i][k]->set_current_density_entry(density_matrix_.ref(i,k));
      
    } // for k
    
    // sort significant_nu_for_mu
    
    std::sort(significant_nu_for_mu_[i], significant_nu_for_mu_[i]+next_ind, 
              Link::Prescreening_Sort_);
              
    
    /*
    for (index_t a = 0; a < next_ind; a++) {
      printf("sort_val: %g\n", 
             significant_nu_for_mu_[i][a]->max_schwartz_factor() * 
             significant_nu_for_mu_[i][a]->current_density_entry());
      
    }
    */
        
    // sort significant_sigma_for_nu_[i] for all i
    // should I do this here?  
    std::sort(significant_sigma_for_nu_[i], 
              significant_sigma_for_nu_[i]+num_significant_sigma_for_nu_[i], 
              Link::ShellPairSort_);
    
  } // for i
  
}


void Link::ComputeExchangeMatrix() {


  // Prescreening loop
  
  PrescreeningLoop_();

  
  // loop over bra shell pairs
  
  for (index_t shell_pair_ind = 0; shell_pair_ind < num_shell_pairs_; 
       shell_pair_ind++) {
       
    ShellPair mu_lambda = shell_pair_list_[shell_pair_ind];
    
    index_t mu_ind = mu_lambda.M_index();
    
    index_t lambda_ind = mu_lambda.N_index();
  
    // loop over nu corresponding to mu
    // what is num_nu? 
    index_t num_nu = num_significant_nu_for_mu_[mu_ind];
    for (index_t sorted_nu_ind = 0; sorted_nu_ind < num_nu; sorted_nu_ind++) {
    
      index_t significant_sigmas = 0;
      
      // need to get this from the first sorted list
      BasisShell* nu_shell = significant_nu_for_mu_[mu_ind][sorted_nu_ind];
      // not sure this will be right for higher momenta
      index_t nu_ind = nu_shell->start_index();
    
      index_t num_sigma = num_significant_sigma_for_nu_[nu_ind];

      for (index_t sigma_ind = 0; sigma_ind < num_sigma; sigma_ind++) {
      
        ShellPair* nu_sigma = significant_sigma_for_nu_[nu_ind][sigma_ind];
        
        if (abs(density_matrix_.ref(mu_ind, nu_ind)) * mu_lambda.schwartz_factor() * 
            nu_sigma->schwartz_factor() > threshold_) {
      
          // store or compute the eri
          
          double integral = eri::ComputeShellIntegrals(mu_lambda, *nu_sigma);
          // contract with mu, nu; mu, sigma; lambda, nu; lambda, sigma
          // sum into lambda, sigma; lambda, nu; mu, sigma; mu, nu
          
          double mu_nu_int = density_matrix_.ref(mu_ind, nu_ind) * integral;
          double mu_sigma_int = density_matrix_.ref(mu_ind, sigma_ind) * integral;
          double lambda_nu_int = density_matrix_.ref(lambda_ind, nu_ind) * integral;
          double lambda_sigma_int = density_matrix_.ref(lambda_ind, sigma_ind) * integral;
          
          double mu_nu_exc = exchange_matrix_.ref(mu_ind, nu_ind);
          double mu_sigma_exc = exchange_matrix_.ref(mu_ind, sigma_ind);
          double lambda_nu_exc = exchange_matrix_.ref(lambda_ind, nu_ind);
          double lambda_sigma_exc = exchange_matrix_.ref(lambda_ind, sigma_ind);
          
          exchange_matrix_.set(mu_ind, nu_ind, mu_nu_exc + lambda_sigma_int);
          exchange_matrix_.set(mu_ind, sigma_ind, mu_sigma_exc + lambda_nu_int);
          exchange_matrix_.set(lambda_ind, nu_ind, 
                               lambda_nu_exc + mu_sigma_int);
          exchange_matrix_.set(lambda_ind, sigma_ind, 
                               lambda_sigma_exc + mu_nu_int);
          
          significant_sigmas++;
          
        }
        // leave sigma loop, since it is sorted
        else {
          break;
        }
        
      } // for sigma_ind
      
      // exit loop if this nu has no sigmas that count
      if (significant_sigmas == 0) {
        break;
      }
    
    } // for nu_ind

    // loop over nu corresponding to lambda and do same
    // why is this necessary as well?  
    // I think it's important for the symmetry, lambda mu never shows up as a 
    // shell pair
    // shouldn't do this if lambda == mu

    if (mu_ind != lambda_ind) {
      num_nu = num_significant_nu_for_mu_[lambda_ind];
      
      for (index_t sorted_nu_ind = 0; sorted_nu_ind < num_nu; sorted_nu_ind++) {
        
        index_t significant_sigmas = 0;
        
        // need to get this from the first sorted list
        BasisShell* nu_shell = significant_nu_for_mu_[mu_ind][sorted_nu_ind];
        // not sure this will be right for higher momenta
        index_t nu_ind = nu_shell->start_index();
        
        index_t num_sigma = num_significant_sigma_for_nu_[nu_ind];
        
        for (index_t sigma_ind = 0; sigma_ind < num_sigma; sigma_ind++) {
          
          ShellPair* nu_sigma = significant_sigma_for_nu_[nu_ind][sigma_ind];
          
          if (abs(density_matrix_.ref(mu_ind, nu_ind)) * mu_lambda.schwartz_factor() * 
              nu_sigma->schwartz_factor() > threshold_) {
            
            // store or compute the eri
            double integral = eri::ComputeShellIntegrals(mu_lambda, *nu_sigma);
            // contract with mu, nu; mu, sigma; lambda, nu; lambda, sigma
            // sum into lambda, sigma; lambda, nu; mu, sigma; mu, nu
            
            double mu_nu_int = density_matrix_.ref(mu_ind, nu_ind) * integral;
            double mu_sigma_int = density_matrix_.ref(mu_ind, sigma_ind) * integral;
            double lambda_nu_int = density_matrix_.ref(lambda_ind, nu_ind) * integral;
            double lambda_sigma_int = density_matrix_.ref(lambda_ind, sigma_ind) * integral;
            
            double mu_nu_exc = exchange_matrix_.ref(mu_ind, nu_ind);
            double mu_sigma_exc = exchange_matrix_.ref(mu_ind, sigma_ind);
            double lambda_nu_exc = exchange_matrix_.ref(lambda_ind, nu_ind);
            double lambda_sigma_exc = exchange_matrix_.ref(lambda_ind, sigma_ind);
            
            exchange_matrix_.set(mu_ind, nu_ind, mu_nu_exc + lambda_sigma_int);
            exchange_matrix_.set(mu_ind, sigma_ind, mu_sigma_exc + lambda_nu_int);
            exchange_matrix_.set(lambda_ind, nu_ind, 
                                 lambda_nu_exc + mu_sigma_int);
            exchange_matrix_.set(lambda_ind, sigma_ind, 
                                 lambda_sigma_exc + mu_nu_int);

            
            significant_sigmas++;
            
          }
          // leave sigma loop, since it is sorted
          else {
            break;
          }
          
        } // for sigma_ind
        
        // exit loop if this nu has no sigmas that count
        if (significant_sigmas == 0) {
          break;
        }
        
      } // for nu_ind
    
    } // if lambda != mu


    // compute significant integrals
    // may just do this inside loop


  } // for bra shell pairs

  
} // ComputeExchangeMatrix()



void Link::OutputExchangeMatrix(Matrix* exc_out) {

  exc_out->Copy(exchange_matrix_);
  
} // OutputFockMatrix

