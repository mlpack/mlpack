#include "cfmm_coulomb.h"

void CFMMCoulomb::ComputeCharges_() {
  
  // this loop should be combined with the one above
  for (index_t i = 0; i < num_shell_pairs_; i++) {
    
    // Needs to become the sum of the relevant density entries for the shell
    // pair, then the charge is the charge of the entire ShellPair treated 
    // as a single point charge.  
    /*
    index_t m_ind = shell_pairs_[i].M_index();
    index_t n_ind = shell_pairs_[i].N_index();
    double new_charge = density_.ref(m_ind, n_ind);
    */
    
    double new_charge = 0.0;
    for (index_t a = 0; a < shell_pairs_[i].M_Shell()->num_functions(); a++) {
      for (index_t b = 0; b < shell_pairs_[i].N_Shell()->num_functions(); b++) {
        new_charge += density_.ref(shell_pairs_[i].M_Shell()->matrix_index(a),
                                   shell_pairs_[i].N_Shell()->matrix_index(b))
                      * shell_pairs_[i].M_Shell()->normalization_constant(a)
                      * shell_pairs_[i].N_Shell()->normalization_constant(b);
      }
    }
    
    // Lumping the coefficients of the integral into the charges
    // this may include too many factors of pi over gamma
    new_charge *= shell_pairs_[i].overlap();
    //new_charge /= pow(shell_pairs_[i].exponent(), 1.5);
    //new_charge *= shell_pairs_[i].M_Shell()->normalization_constant();
    //new_charge *= shell_pairs_[i].N_Shell()->normalization_constant();
    
    // If not on diagonal, then needs to be counted twice
    if (shell_pairs_[i].M_Shell() != shell_pairs_[i].N_Shell()) {
      new_charge *= 2;
    }
    
    charges_.set(0, i, new_charge);
    
  } // for i
  
}

void CFMMCoulomb::ScreenCharges_() {

  num_shell_pairs_ = eri::ComputeShellPairs(&shell_pairs_, shells_, 
                                            charge_thresh_, density_);
  
  charges_.Init(1,num_shell_pairs_);
       
  fx_result_int(mod_, "num_shell_pairs", num_shell_pairs_);
  fx_result_int(mod_, "num_shell_pairs_screened", 
                num_shells_ * (num_shells_ - 1) / 2);
                                            
  charge_centers_.Init(3, num_shell_pairs_);
  charge_exponents_.Init(1, num_shell_pairs_);

  for (index_t i = 0; i < num_shell_pairs_; i++) {
    
    // add to output matrix
    charge_centers_.CopyVectorToColumn(i, shell_pairs_[i].center());
    
    charge_exponents_.set(0,i, shell_pairs_[i].exponent());
    
  } // for i
  
  
  ComputeCharges_();
  
} // ScreenCharges()


void CFMMCoulomb::MultipoleInit_() {

  fx_module* multipole_mod = fx_submodule(mod_, "multipole_cfmm");
  
  ArrayList<ShellPair*> shell_pairs_pointers;
  shell_pairs_pointers.Init(num_shell_pairs_);
  for (int i = 0; i < num_shell_pairs_; i++) {
    shell_pairs_pointers[i] = &(shell_pairs_[i]);
  }

  cfmm_algorithm_.Init(charge_centers_, charge_centers_, charges_, 
                       charge_exponents_, true, multipole_mod, &coulomb_mat_, 
                       shell_pairs_pointers, &density_);
                       
  multipole_init_called_ = true;

} // MultipoleInit()

void CFMMCoulomb::MultipoleComputation_() {

  fx_timer_start(mod_, "multipole_computation");
  cfmm_algorithm_.Compute(&multipole_output_);
  fx_timer_stop(mod_, "multipole_computation");

} // MultipoleComputation

double CFMMCoulomb::NaiveBaseCase_(Vector& q_col, double q_band, Vector& r_col, 
                                  double r_band, double r_charge) {

  double potential = 0.0;

  double sq_dist = la::DistanceSqEuclidean(q_col, r_col);
  double dist = sqrt(sq_dist);
  double erf_argument = sqrt(q_band * r_band / (q_band + r_band));
  
  // This implements the kernel function used for the base case
  // in the page 2 of the CFMM paper...
  
  if(dist > 0) {
    potential += r_charge * erf(erf_argument * dist) / dist;
  }
  else {
    // F_0 needs to be 1, not sqrt(pi)/2
    potential += r_charge * erf_argument * 2 / sqrt(math::PI);
  }
  
  return potential;
  
}

void CFMMCoulomb::NaiveComputation_() {
  
  /*
  multipole_output_.Destruct();
  cfmm_algorithm_.NaiveCompute(&multipole_output_);
   */
   
  multipole_output_.Destruct();
  multipole_output_.Init(num_shell_pairs_);
  multipole_output_.SetZero();
  
  //printf("doing naive\n");
  
  for(index_t q = 0; q < num_shell_pairs_; q++) {
  
    Vector q_col;
    charge_centers_.MakeColumnVector(q, &q_col);
    
    double q_band = charge_exponents_.ref(0, q);
  
    for (index_t r = 0; r < num_shell_pairs_; r++) {
    
      Vector r_col;
      charge_centers_.MakeColumnVector(r, &r_col);
      double r_band = charge_exponents_.ref(0, r);
      
      double r_charge = charges_.ref(0, r);
    
      multipole_output_[q] += NaiveBaseCase_(q_col, q_band, r_col, r_band, 
                                             r_charge);
    
    }
  
  }
  
}


void CFMMCoulomb::MultipoleCleanup_() {

  //coulomb_mat_.PrintDebug("J (before far field)");
  
  for (index_t i = 0; i < num_shell_pairs_; i++) {
    
    //index_t m_ind = shell_pairs_[i].M_index();
    //index_t n_ind = shell_pairs_[i].N_index();
    
    // multiply by prefactors and such
    
    
    double coulomb_entry = multipole_output_[i];
    
    // Need to account for the fact that base case contributions already 
    // include this
    coulomb_entry *= shell_pairs_[i].overlap();
    //coulomb_entry /= pow(shell_pairs_[i].exponent(), 1.5);
    // extra square root of pi/2 from erf
    //coulomb_entry *= pow(math::PI, 3.0);
    //coulomb_entry *= shell_pairs_[i].M_Shell()->normalization_constant();
    //coulomb_entry *= shell_pairs_[i].N_Shell()->normalization_constant();
    
    Matrix this_mat;
    this_mat.Init(shell_pairs_[i].M_Shell()->num_functions(),
                  shell_pairs_[i].N_Shell()->num_functions());
    
    for (index_t a = 0; a < shell_pairs_[i].M_Shell()->num_functions(); a++) {
      for (index_t b = 0; b < shell_pairs_[i].N_Shell()->num_functions(); b++) {
        
        this_mat.set(a, b, coulomb_entry 
                           * shell_pairs_[i].M_Shell()->normalization_constant(a) 
                           * shell_pairs_[i].N_Shell()->normalization_constant(b));
        
      }
    }
    
    
    eri::AddSubmatrix(shell_pairs_[i].M_Shell()->matrix_indices(),
                      shell_pairs_[i].N_Shell()->matrix_indices(),
                      this_mat, &coulomb_mat_);
    
    if (shell_pairs_[i].M_Shell() != shell_pairs_[i].N_Shell()) {
      
      Matrix this_trans;
      la::TransposeInit(this_mat, &this_trans);
      eri::AddSubmatrix(shell_pairs_[i].N_Shell()->matrix_indices(),
                        shell_pairs_[i].M_Shell()->matrix_indices(),
                        this_trans, &coulomb_mat_);
      
    }
    
    //coulomb_mat_.set(m_ind, n_ind, coulomb_entry);
    //coulomb_mat_.set(n_ind, m_ind, coulomb_entry);
    
  } // for i
  

} // MultipoleCleanup


/////////////////// Public Functions ///////////////////////////

void CFMMCoulomb::Compute() {

  printf("====CFMM Compute====\n");
  
  fx_timer_start(mod_, "cfmm_time");
  
  MultipoleComputation_();
  
  if (fx_param_exists(mod_, "do_naive")) {
    NaiveComputation_();
  }
  
  MultipoleCleanup_();

  fx_timer_stop(mod_, "cfmm_time");

   
} // ComputeCoulomb()


void CFMMCoulomb::OutputCoulomb(Matrix* coulomb_out) {

  coulomb_out->Copy(coulomb_mat_);

} // OutputCoulomb()


void CFMMCoulomb::UpdateDensity(const Matrix& new_density) {
  
  density_.CopyValues(new_density);
  
  // need to update charges
  ComputeCharges_();
  
  printf("cfmm update density\n");
  
  // not sure if this is the right way to destruct this
  if (multipole_init_called_) {
    printf("destroying ContinuousFmm\n");
    //cfmm_algorithm_.~ContinuousFmm();
    //delete cfmm_algorithm_;
    //cfmm_algorithm_.Destruct();
    //cfmm_algorithm_.Reset(charges_);
  }
  
}
