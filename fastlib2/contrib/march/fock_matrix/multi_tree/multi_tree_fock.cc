#include "multi_tree_fock.h"

double MultiTreeFock::NodesMaxIntegral_(FockTree* mu, FockTree* nu, 
                                        FockTree* rho, FockTree* sigma) {

  double integral;
  
  double rho_sigma_min_dist = rho->bound().MinDistanceSq(sigma->bound());
  
  double mu_nu_min_dist = mu->bound().MinDistanceSq(nu->bound());
  
  // compute the average boxes
  DHrectBound<2> mu_nu_ave;
  mu_nu_ave.AverageBoxesInit(mu->bound(), nu->bound());
  DHrectBound<2> rho_sigma_ave;
  rho_sigma_ave.AverageBoxesInit(rho->bound(), sigma->bound());
  
  double four_way_min_dist = mu_nu_ave.MinDistanceSq(rho_sigma_ave);

  double mu_max_band = mu->stat().max_bandwidth();
  double nu_max_band = nu->stat().max_bandwidth();
  double rho_max_band = rho->stat().max_bandwidth();
  double sigma_max_band = sigma->stat().max_bandwidth();
  
  // Double check that the maximum integral comes from the maximum bandwidths
  // Need to normalize
  // Make this as efficient as possible
  integral = eri::DistanceIntegral(mu_max_band, nu_max_band, 
                                   rho_max_band, sigma_max_band, 
                                   mu_nu_min_dist, rho_sigma_min_dist, 
                                   four_way_min_dist);
  
  return integral;

} // NodesMaxIntegral_()

double MultiTreeFock::NodesMinIntegral_(FockTree* mu, FockTree* nu, 
                                        FockTree* rho, FockTree* sigma) {
   
  double integral;
  
  double rho_sigma_max_dist = rho->bound().MaxDistanceSq(sigma->bound());
  
  double mu_nu_max_dist = mu->bound().MaxDistanceSq(nu->bound());
  
  DHrectBound<2> mu_nu_ave;
  mu_nu_ave.AverageBoxesInit(mu->bound(), nu->bound());
  DHrectBound<2> rho_sigma_ave;
  rho_sigma_ave.AverageBoxesInit(rho->bound(), sigma->bound());
  
  double four_way_max_dist = mu_nu_ave.MaxDistanceSq(rho_sigma_ave);
  
  double mu_min_band = mu->stat().min_bandwidth();
  double nu_min_band = nu->stat().min_bandwidth();
  double rho_min_band = rho->stat().min_bandwidth();
  double sigma_min_band = sigma->stat().min_bandwidth();
  
  // Double check that the maximum integral comes from the maximum bandwidths
  // Need to normalize
  // Make this as efficient as possible
  integral = eri::DistanceIntegral(mu_min_band, nu_min_band, 
                                   rho_min_band, sigma_min_band, 
                                   mu_nu_max_dist, rho_sigma_max_dist, 
                                   four_way_max_dist);
  
  return integral;
  
} // NodesMinIntegral_()

/**
 * The centroid approximation of the interaction between these nodes
 */
double MultiTreeFock::NodesMidpointIntegral_(FockTree* mu, FockTree* nu, 
                                             FockTree* rho, FockTree* sigma) {

  Vector mu_center;
  mu->bound().CalculateMidpoint(&mu_center);
  Vector nu_center;
  nu->bound().CalculateMidpoint(&nu_center);
  Vector rho_center;
  rho->bound().CalculateMidpoint(&rho_center);
  Vector sigma_center;
  sigma->bound().CalculateMidpoint(&sigma_center);
  
  double mu_max_band = mu->stat().max_bandwidth();
  double nu_max_band = nu->stat().max_bandwidth();
  double rho_max_band = rho->stat().max_bandwidth();
  double sigma_max_band = sigma->stat().max_bandwidth();  
  
  double mu_min_band = mu->stat().min_bandwidth();
  double nu_min_band = nu->stat().min_bandwidth();
  double rho_min_band = rho->stat().min_bandwidth();
  double sigma_min_band = sigma->stat().min_bandwidth();
  
  double mu_mid_band = 0.5*(mu_max_band + mu_min_band);
  double nu_mid_band = 0.5*(nu_max_band + nu_min_band);
  double rho_mid_band = 0.5*(rho_max_band + rho_min_band);
  double sigma_mid_band = 0.5*(sigma_max_band + sigma_min_band);
  
  // not sure the SSSS function is right here
  // will have to be changed for p-integrals anyway
  double integral = eri::SSSSIntegral(mu_mid_band, mu_center, nu_mid_band, 
                                      nu_center, rho_mid_band, rho_center, 
                                      sigma_mid_band, sigma_center); 
  
  return integral;

} // NodesMidpointIntegral_()


bool DualTreeIntegrals::RectangleOnDiagonal_(FockTree* mu, 
                                             FockTree* nu) {
  
  if (mu == nu) {
    return false;
  }
  else if (mu->begin() >= nu->end()){
    return false;
  }
  else {
    return true;
  }

} // RectangleOnDiagonal_  


index_t MultiTreeFock::CountOnDiagonal_(SquareFockTree* rho_sigma) {
  
  index_t on_diagonal;
  
  // one of these should be square, which one?
  SquareFockTree* left_child = rho_sigma->left();
  SquareFockTree* right_child = rho_sigma->right();
  
  if (left_child->query1() == left_child->query2()) {
    on_diagonal = left_child->query1()->count() * left_child->query2()->count();
  }
  else {
    DEBUG_ASSERT(right_child->query1() == right_child->query2());
    on_diagonal = right_child->query1()->count() * 
      right_child->query2()->count();
  }
  
  return on_diagonal;                                   
  
} // CountOnDiagonal_()


bool MultiTreeFock::CanApproximateCoulomb_(SquareTree* mu_nu, 
                                           SquareTree* rho_sigma, 
                                           double* approx_val) {

  IntegralTree* mu = mu_nu->query1();
  IntegralTree* nu = mu_nu->query2();
  IntegralTree* rho = rho_sigma->query1();
  IntegralTree* sigma = rho_sigma->query2();
  
  bool can_prune = false;
  
  double up_bound = NodesMaxIntegral_(mu, nu, rho, sigma);
                    
  double low_bound = NodesMinIntegral_(mu, nu, rho, sigma);
  
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
  
  double approx_val = NodesMidpointIntegral_(mu, nu, rho, sigma);
  
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


bool MultiTreeFock::CanApproximateExchange_(SquareIntegralTree* mu_nu, 
                                            SquareIntegralTree* rho_sigma, 
                                            double* approximate_value) { 
                                            
  IntegralTree* mu = mu_nu->query1();
  IntegralTree* nu = mu_nu->query2();
  IntegralTree* rho = rho_sigma->query1();
  IntegralTree* sigma = rho_sigma->query2();
  
  // If it is possible to prune here, that will still be true for the two 
  // children, where we won't have to try to handle the near symmetry
  if (RectangleOnDiagonal_(rho, sigma)) {
    DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
    return false;
  }
  
  
  bool can_prune = false;
  
  double up_bound = NodesMaxIntegral_(mu, rho, nu, sigma);
  
  double low_bound = NodesMinIntegral_(mu, rho, nu, sigma);
  
  Vector mu_center;
  mu->bound().CalculateMidpoint(&mu_center);
  Vector nu_center;
  nu->bound().CalculateMidpoint(&nu_center);
  Vector rho_center;
  rho->bound().CalculateMidpoint(&rho_center);
  Vector sigma_center;
  sigma->bound().CalculateMidpoint(&sigma_center);
  
  double approx_val = NodesMidpointIntegral_(mu, rho, nu, sigma);
  
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
    
  if (rho != sigma) {
    
    double mu_sigma_min_dist = mu->bound().MinDistanceSq(sigma->bound());
    double mu_sigma_max_dist = mu->bound().MaxDistanceSq(sigma->bound());
    
    double nu_rho_min_dist = nu->bound().MinDistanceSq(rho->bound());
    double nu_rho_max_dist = nu->bound().MaxDistanceSq(rho->bound());
    
    DHrectBound<2> mu_sigma_ave;
    mu_sigma_ave.AverageBoxesInit(mu->bound(), sigma->bound());
    DHrectBound<2> nu_rho_ave;
    nu_rho_ave.AverageBoxesInit(nu->bound(), rho->bound());
    
    double mu_sigma_four_way_min_dist = mu_sigma_ave.MinDistanceSq(nu_rho_ave);
    double mu_sigma_four_way_max_dist = mu_sigma_ave.MaxDistanceSq(nu_rho_ave);
    
    double mu_sigma_upper = NodesMaxIntegral_(mu, sigma, nu, rho);
    double mu_sigma_lower = NodesMinIntegral_(mu, sigma, nu, rho);
    double mu_sigma_ave = NodesMidpointIntegral_(mu, sigma, nu, rho);
    
    up_bound = up_bound + mu_sigma_upper;
    
    low_bound = low_bound + mu_sigma_lower;
    
    approx_val = approx_val + mu_sigma_ave;
    
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    
  }
  
  // Because the exchange integrals have an extra 1/2 factor
  up_bound = up_bound * 0.5;
  low_bound = low_bound * 0.5;
  approx_val = approx_val * 0.5;
  
  up_bound = up_bound * rho->count() * sigma->count();
  low_bound = low_bound * rho->count() * sigma->count();
  approx_val = approx_val * rho->count() * sigma->count();
  
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
  
  double old_up_bound = up_bound;
  
  // account for possibly negative density matrix entries
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
  
  // multiply in the density bounds
  // I'm not sure this is right, but it will work for now
  approx_val = approx_val * 0.5 * (rho_sigma->stat().density_upper_bound() + 
                                   rho_sigma->stat().density_lower_bound());
  
  
  // This is absolute error, I'll need to get the lower bound out of the 
  // mu_nu square tree to do relative
  double my_allowed_error = epsilon_exchange_ * rho->count() * sigma->count()
    / (number_of_basis_functions_ * number_of_basis_functions_);
  
  // The total error I'm incurring is the max error for one integral times 
  // the number of approximations I'm making
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
  double my_max_error = max((up_bound - approx_val), 
                            (approx_val - low_bound));
  
  // if using absolute error, then existing code is fine
  
  // For hybrid error 
  // assuming epsilon coulomb is the relative error tolerance
  // took out <= here too
  if ((fabs(mu_nu->stat().entry_upper_bound()) < hybrid_cutoff_) && 
      (fabs(mu_nu->stat().entry_lower_bound()) < hybrid_cutoff_)) {
    
    DEBUG_ASSERT(mu_nu->stat().entry_upper_bound() >= 
                 mu_nu->stat().entry_lower_bound());
    
    // set my allowed_error to be the absolute bound
    my_allowed_error = my_allowed_error * epsilon_exchange_absolute_ / 
      epsilon_exchange_;
    
  }
  else {
    my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();
  }
  
  // For relative error
  /*my_allowed_error = my_allowed_error * mu_nu->stat().entry_lower_bound();*/
  
  DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
  if (my_max_error < my_allowed_error) {
    
    can_prune = true;
    
    *approximate_value = approx_val;
    //printf("approx_val = %g\n", *approximate_value);
    
  }
  
  return can_prune;
  
} // CanApproximateExchange_



void ComputeCoulombBaseCase_(SquareFockTree* mu_nu, SquareFockTree* rho_sigma) {
                             
                             
                             
                             
}


void ComputeExchangeBaseCase_(SquareFockTree* mu_nu, 
                              SquareFockTree* rho_sigma) {
 
                              
}


///////////////////// public functions ////////////////////////////////////

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