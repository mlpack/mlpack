#include "multi_tree_fock.h"


bool CanApproximateCoulomb_(SquareTree* mu_nu, SquareTree* rho_sigma, 
                            double* approx_val) {

  IntegralTree* mu = mu_nu->query1();
  IntegralTree* nu = mu_nu->query2();
  IntegralTree* rho = rho_sigma->query1();
  IntegralTree* sigma = rho_sigma->query2();
  
  bool can_prune = false;
  
  double rho_sigma_min_dist = rho->bound().MinDistanceSq(sigma->bound());
  double rho_sigma_max_dist = rho->bound().MaxDistanceSq(sigma->bound());
  //double rho_sigma_mid_dist = rho->bound().MidDistanceSq(sigma->bound());
  
  double mu_nu_min_dist = mu->bound().MinDistanceSq(nu->bound());
  double mu_nu_max_dist = mu->bound().MaxDistanceSq(nu->bound());
  //double mu_nu_mid_dist = mu->bound().MidDistanceSq(nu->bound());
  
  
  DHrectBound<2> mu_nu_ave;
  mu_nu_ave.AverageBoxesInit(mu->bound(), nu->bound());
  DHrectBound<2> rho_sigma_ave;
  rho_sigma_ave.AverageBoxesInit(rho->bound(), sigma->bound());
  //ot::Print(mu_nu_ave);
  //ot::Print(rho_sigma_ave);
  
  double four_way_min_dist = mu_nu_ave.MinDistanceSq(rho_sigma_ave);
  double four_way_max_dist = mu_nu_ave.MaxDistanceSq(rho_sigma_ave);
  //double four_way_mid_dist = mu_nu_ave.MidDistanceSq(rho_sigma_ave);
  //printf("four_way_min: %g\n", four_way_min_dist);
  
  double mu_max_band = mu->stat().max_bandwidth();
  double mu_min_band = mu->stat().min_bandwidth();
  double nu_max_band = nu->stat().max_bandwidth();
  double nu_min_band = nu->stat().min_bandwidth();
  double rho_max_band = rho->stat().max_bandwidth();
  double rho_min_band = rho->stat().min_bandwidth();
  double sigma_max_band = sigma->stat().max_bandwidth();
  double sigma_min_band = sigma->stat().min_bandwidth();
  
  // Double check that the maximum integral comes from the maximum bandwidths
  // Need to normalize
  // Make this as efficient as possible
  double up_bound = eri::DistanceIntegral(mu_max_band, nu_max_band, 
                                          rho_max_band, sigma_max_band, 
                                          mu_nu_min_dist, rho_sigma_min_dist, 
                                          four_way_min_dist);
                    
  double low_bound = eri::DistanceIntegral(mu_min_band, nu_min_band, 
                                           rho_min_band, sigma_min_band, 
                                           mu_nu_max_dist, rho_sigma_max_dist, 
                                           four_way_max_dist);
  
  // Need to account for the change in bounds if the density matrix entries 
  // are negative.  If the density lower bound is negative, then the lower 
  // bound becomes the largest integral times the density lower bound, 
  // instead of the smallest integral and vice versa.  
  double old_up_bound = up_bound;
  if (rho_sigma->stat().density_upper_bound() >= 0.0) {
    up_bound = up_bound * rho_sigma->stat().density_upper_bound();
  }
  else {
    up_bound = low_bound * rho_sigma->stat().density_upper_bound();
  }
  if (rho_sigma->stat().density_lower_bound() >= 0.0) {
    low_bound = low_bound * rho_sigma->stat().density_lower_bound();
  }
  else {
    low_bound = old_up_bound * rho_sigma->stat().density_lower_bound();
  }
  
  DEBUG_ASSERT(up_bound >= low_bound);
  
  
  Vector mu_center;
  mu->bound().CalculateMidpoint(&mu_center);
  Vector nu_center;
  nu->bound().CalculateMidpoint(&nu_center);
  Vector rho_center;
  rho->bound().CalculateMidpoint(&rho_center);
  Vector sigma_center;
  sigma->bound().CalculateMidpoint(&sigma_center);
  
  double mu_mid_band = 0.5*(mu_max_band + mu_min_band);
  double nu_mid_band = 0.5*(nu_max_band + nu_min_band);
  double rho_mid_band = 0.5*(rho_max_band + rho_min_band);
  double sigma_mid_band = 0.5*(sigma_max_band + sigma_min_band);
  
  // Change this to include bandwidths
  double approx_val = ComputeSingleIntegral_(mu_center, nu_center, rho_center, 
                                             sigma_center);
  
  // Need to make this work with on diagonal non-square boxes
  if (RectangleOnDiagonal_(rho, sigma)) {
    index_t on_diagonal = CountOnDiagonal_(rho_sigma);
    index_t off_diagonal = (rho->count() * sigma->count()) - on_diagonal;
    up_bound = (on_diagonal * up_bound) + (2 * off_diagonal * up_bound);
    // These two were missing, I think they needed to be here
    low_bound = (on_diagonal * low_bound) + (2 * off_diagonal * low_bound);
    approx_val = (on_diagonal * approx_val) + (2 * off_diagonal * approx_val);
  }
  else if (rho != sigma) {
    up_bound = 2 * up_bound * rho->count() * sigma->count();
    low_bound = 2 * low_bound * rho->count() * sigma->count();
    approx_val = 2 * approx_val * rho->count() * sigma->count();
  }
  else {
    up_bound = up_bound * rho->count() * sigma->count();
    low_bound = low_bound * rho->count() * sigma->count();
    approx_val = approx_val * rho->count() * sigma->count();
  }
  
  // I'm not sure this is right, but it will work for now
  approx_val = approx_val * 0.5 * (rho_sigma->stat().density_upper_bound() + 
                                   rho_sigma->stat().density_lower_bound());
  
  
  double my_allowed_error = epsilon_coulomb_ * rho->count() * sigma->count()
    / (number_of_basis_functions_ * number_of_basis_functions_);
  
  // The total error I'm incurring is the max error for one integral times 
  // the number of approximations I'm making
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
  double my_max_error = max((up_bound - approx_val), 
                            (approx_val - low_bound));
  
  // if using absolute error, then existing code is fine
  
  bool below_cutoff;
  
  // For hybrid error 
  // assuming epsilon coulomb is the relative error tolerance
  // took out the <=, just in case
  // I think this is wrong, since the upper and lower bounds are initialized 
  // at 0.0, this means it always prunes with absolute error
  if ((fabs(mu_nu->stat().entry_upper_bound()) < hybrid_cutoff_) && 
      (fabs(mu_nu->stat().entry_lower_bound()) < hybrid_cutoff_)) {
    
    // set my allowed_error to be the absolute bound
    my_allowed_error = my_allowed_error * epsilon_coulomb_absolute_ / 
    epsilon_coulomb_;
    
    below_cutoff = true;
    
  }
  else {
    my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();
    below_cutoff = false;
  }
  
  // For relative error
  /*my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();*/
  
  DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
  
  if (my_max_error < my_allowed_error) {
    
    can_prune = true;
    
    *approximate_value = approx_val;
    //printf("approx_val = %g\n", *approximate_value);
    
    if (below_cutoff) {
      num_absolute_prunes_++;
    }
    else {
      num_relative_prunes_++;
    }
    
  }
  
  return can_prune;

}


bool CanApproximateExchange_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma, 
                             double* approx_val) {
    
  
  
}

void ComputeCoulombBaseCase_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma) {
                             
                             
                             
                             
}


void ComputeExchangeBaseCase_(SquareFockTree* mu_nu, 
                              SquareFockTree* rho_sigma) {
 
                              
}

void ComputeFockMatrix() {

  fx_timer_start(module_, "coulomb_recursion");
  ComputeCoulombRecursion_(square_tree_, square_tree_);  
  fx_timer_stop(module_, "coulomb_recursion");

  // Will need to be followed by clearing the tree and computing the exchange 
  // matrix
  // I think this is the only resetting the tree will need
  SetEntryBounds_();
  ResetTreeForExchange_(square_tree_);
  fx_timer_start(module_, "exchange_recursion");
  ComputeExchangeRecursion_(square_tree_, square_tree_);
  fx_timer_stop(module_, "exchange_recursion");
    
  la::SubOverwrite(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  

} // ComputeFockMatrix()