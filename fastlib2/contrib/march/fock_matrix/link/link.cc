#include "link.h"


bool Link::Prescreening_Sort_(BasisShell* shellA, BasisShell* shellB) {

  return ((shellA->max_schwartz_factor() * shellA->current_density_entry()) < 
          (shellB->max_schwartz_factor() * shellB->current_density_entry()));

}

bool Link::ShellPairSort_(BasisShell* shellA, BasisShell* shellB) {

  return(shellA);

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
      if (density_mat_.ref(i,j) * shell_max_[i] * shell_max_[j] > threshold_) {
        
        // Store significant j for each i
        // What is the best way to do this?  This list will have to be sorted
        // This fails if there are more than num_shells_/2 significant elements
        significant_nu_index[next_ind] = j;
        
        // I need to store next_ind somehow in order to be able to iterate 
        // later
        next_ind++;
        
      }
      
    } // for j
    
    num_significant_mu_pairs_[i] = next_ind;
    
    significant_mu_pairs_[i] = (BasisShell**)malloc(next_ind * 
                                                    sizeof(BasisShell**)); 
    //BasisShell** significant_mu_pairs_[i] = significant_mu_pairs_[i];
    
    
    for (index_t k = 0; k < next_ind; k++) {
      
      significant_mu_pairs_[i][k] = shell_list_.begin() + 
      significant_nu_index[k];
      significant_mu_pairs_[i][k]->set_max_schwartz_factor(shell_max_[k]);
      significant_mu_pairs_[i][k]->set_current_density_entry(density_mat_.ref(i,k));
      
    } // for k
    
    // sort significant_nu_for_mu
    
    std::sort(significant_mu_pairs_[i], significant_mu_pairs_[i]+next_ind, 
              Link::Prescreening_Sort_);
    
    
    //ot::Print(significant_mu_pairs_[i]);
    for (index_t a = 0; a < next_ind; a++) {
      printf("sort_val: %g\n", 
             significant_mu_pairs_[i][a]->max_schwartz_factor() * 
             significant_mu_pairs_[i][a]->current_density_entry());
      
    }
    
    // sort significant_sigma_for_nu_[i] for all i
      
    std::sort(significant_sigma_for_nu_[i], 
              significant_sigma_for_nu_[i]+num_significant_sigma_for_nu_[i], 
              Link::ShellPairSort_);
    
  } // for i
  
}

void Link::SortShellPairLists_() {

/*  index_t shell_pair_ind = 0;

  while(shell_pair_ind < num_shell_pairs_) {
  
    ShellPair this_pair = shell_pairs_[shell_pair_ind];
    
    index_t m_ind = this_pair.m_ind();
    index_t n_ind = this_pair.n_ind();
    
    while(cur_ind == m_ind) {
    
      // add to list and counter
      
      
      
    
    } // while cur_ind
    
    shell_pair_ind++;
  
  } // while shell_pair_ind

*/

  for (index_t shell_pair_ind = 0; shell_pair_ind < num_shell_pairs_; 
       shell_pair_ind++) { 
  
    ShellPair this_pair = shell_pairs_[shell_pair_ind];
    
    index_t m_ind = this_pair.m_ind();
    index_t n_ind = this_pair.n_ind();
    
    // don't know how much space to allocate
    significant_sigma_for_nu_[m_ind][next_ind] = x;
  
  } // for shell_pair_ind

} // SortShellPairLists_()

void Link::ComputeExchangeMatrix() {


  // Prescreening loop
  
  PrescreeningLoop_();

  
  // loop over bra shell pairs
  
  for (index_t shell_pair_ind = 0; shell_pair_ind < num_shell_pairs_; 
       shell_pair_ind++) {
       
    ShellPair mu_lambda = shell_pair_list_[shell_pair_ind];
    BasisShell mu_shell = mu_lambda.M_Shell();
    BasisShell lambda_shell = mu_lambda.N_Shell();
    
    index_t mu_ind = mu_lambda.M_index();
    
    index_t lambda_ind = mu_lambda.N_index();
  
    // loop over nu corresponding to mu
    // what is num_nu? 
    index_t num_nu = num_significant_mu_pairs_[mu_ind];
    for (index_t nu_ind = 0; nu_ind < num_nu; nu_ind++) {
    
      BasisShell nu_shell = *(significant_mu_pairs_[mu_ind][nu_ind]);
      
      // store how many sigificant sigmas for this nu in order to exit loop
      index_t significant_sigmas = 0;
      
      // loop over significant sigmas

      index_t num_sigma = num_shells_for_i[nu_ind];

      for (index_t sigma_ind = 0; sigma_ind < num_sigma; sigma_ind++) {
      
        // need to store the sigma shells for nu somewhere
        // I think they need to be sorted
        BasisShell sigma_shell = what;
      
        // fill in these
        // would be nice to be working with shell pairs, since they store the 
        // Schwartz factors
        if (abs(density_mat_.ref(mu, nu)) * mu_lambda_schwartz * nu_sigma_schwartz > cutoff) {
      
          // store nu sigma as a significant shell pair to be computed
          // is it really necessary to store these, or can I compute it now?
          // I think for my simple integral code it won't make a difference
        
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




  // compute significant integrals





  } // for bra shell pairs

  
} // ComputeFockMatrix()



void Link::OutputExchangeMatrix(Matrix* exc_out) {

  exc_out->Copy(exchange_matrix_);
  
} // OutputFockMatrix