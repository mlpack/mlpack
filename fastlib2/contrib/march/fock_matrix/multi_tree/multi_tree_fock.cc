#include "multi_tree_fock.h"

double MultiTreeFock::NodeMaxNorm_(FockTree* mu) {

  double norm = pow(2.0/math::PI, 0.75);
  // add momentum here
  norm *= pow(mu->stat().max_bandwidth(), 0.75);

  return norm;

}

double MultiTreeFock::NodeMinNorm_(FockTree* mu) {

  double norm = pow(2.0/math::PI, 0.75);
  // add momentum here
  norm *= pow(mu->stat().min_bandwidth(), 0.75);
  
  return norm;

}

double MultiTreeFock::NodeAveNorm_(FockTree* mu) {

  double norm = pow(2.0/math::PI, 0.75);
  // add momentum here
  norm *= pow((0.5*(mu->stat().max_bandwidth() + mu->stat().min_bandwidth())), 
              0.75);
  
  return norm;
  
}

double MultiTreeFock::NodesMaxIntegral_(SquareTree* mu_nu, SquareTree* rho_sigma) {

  double integral = 2 * pow_pi_2point5_;
  
  double mu_nu_dist = mu_nu->query1()->bound().MinDistanceSq(mu_nu->query2()->bound());
  double rho_sigma_dist = rho_sigma->query1()->bound().MinDistanceSq(rho_sigma->query2()->bound());
  
  integral *= mu_nu->stat().max_gpt_factor();
  integral *= rho_sigma->stat().max_gpt_factor();
  
  integral *= mu_nu->query1()->stat().max_normalization();
  integral *= mu_nu->query2()->stat().max_normalization();
  integral *= rho_sigma->query1()->stat().max_normalization();
  integral *= rho_sigma->query2()->stat().max_normalization();

  integral /= (mu_nu->stat().min_gamma() * rho_sigma->stat().min_gamma() * 
               (mu_nu->stat().min_gamma() + rho_sigma->stat().min_gamma()));
               
  double four_way_dist_sq = mu_nu->bound().MinDistanceSq(rho_sigma->bound());
  
  /*
  printf("Min Dists: mu_nu = %g, rho_sigma = %g, four_way = %g\n",
         mu_nu_dist, rho_sigma_dist, four_way_dist_sq);
  */
  
  integral *= eri::IntegralMomentumFactor(mu_nu->stat().min_gamma(), 
                                          rho_sigma->stat().min_gamma(), 
                                          four_way_dist_sq);

  return integral;

} // NodesMaxIntegral(SquareTree)

double MultiTreeFock::NodesMinIntegral_(SquareTree* mu_nu, SquareTree* rho_sigma) {
  
  double integral = 2 * pow_pi_2point5_;
  
  double mu_nu_dist = mu_nu->query1()->bound().MaxDistanceSq(mu_nu->query2()->bound());
  double rho_sigma_dist = rho_sigma->query1()->bound().MaxDistanceSq(rho_sigma->query2()->bound());

  
  integral *= mu_nu->stat().min_gpt_factor();
  integral *= rho_sigma->stat().min_gpt_factor();
  
  integral *= mu_nu->query1()->stat().min_normalization();
  integral *= mu_nu->query2()->stat().min_normalization();
  integral *= rho_sigma->query1()->stat().min_normalization();
  integral *= rho_sigma->query2()->stat().min_normalization();
  
  integral /= (mu_nu->stat().max_gamma() * rho_sigma->stat().max_gamma() * 
               (mu_nu->stat().max_gamma() + rho_sigma->stat().max_gamma()));
  
  double four_way_dist_sq = mu_nu->bound().MaxDistanceSq(rho_sigma->bound());
  
  integral *= eri::IntegralMomentumFactor(mu_nu->stat().max_gamma(), 
                                          rho_sigma->stat().max_gamma(), 
                                          four_way_dist_sq);
  
  /*
  printf("Max Dists: mu_nu = %g, rho_sigma = %g, four_way = %g\n",
         mu_nu_dist, rho_sigma_dist, four_way_dist_sq);
  */
  
  return integral;
  
} // NodesMaxIntegral(SquareTree)



double MultiTreeFock::NodesMaxIntegral_(FockTree* mu, FockTree* nu, 
                                        FockTree* rho, FockTree* sigma) {

  double integral;
  
  double rho_sigma_min_dist = rho->bound().MinDistanceSq(sigma->bound());
  
  double mu_nu_min_dist = mu->bound().MinDistanceSq(nu->bound());
  
  // compute the average boxes
  DHrectBound<2> mu_nu_ave;
  mu_nu_ave.WeightedAverageBoxesInit(mu->stat().min_bandwidth(), mu->stat().max_bandwidth(), 
                                     mu->bound(), nu->stat().min_bandwidth(), 
                                     nu->stat().max_bandwidth(), nu->bound());
  DHrectBound<2> rho_sigma_ave;
  rho_sigma_ave.WeightedAverageBoxesInit(rho->stat().min_bandwidth(), rho->stat().max_bandwidth(), 
                                 rho->bound(), sigma->stat().min_bandwidth(), 
                                 sigma->stat().max_bandwidth(), sigma->bound());
  
  double four_way_min_dist = mu_nu_ave.MinDistanceSq(rho_sigma_ave);

  /*
  double mu_max_band = mu->stat().max_bandwidth();
  double nu_max_band = nu->stat().max_bandwidth();
  double rho_max_band = rho->stat().max_bandwidth();
  double sigma_max_band = sigma->stat().max_bandwidth();
  */

  double mu_min_band = mu->stat().min_bandwidth();
  double nu_min_band = nu->stat().min_bandwidth();
  double rho_min_band = rho->stat().min_bandwidth();
  double sigma_min_band = sigma->stat().min_bandwidth();
  
  // Make this as efficient as possible
  integral = eri::DistanceIntegral(mu_min_band, nu_min_band, 
                                   rho_min_band, sigma_min_band, 
                                   mu_nu_min_dist, rho_sigma_min_dist, 
                                   four_way_min_dist);
                                   
  //integral *= NodeMaxNorm_(mu) * NodeMaxNorm_(nu) * 
  //  NodeMaxNorm_(rho) * NodeMaxNorm_(sigma);
    
  integral *= mu->stat().max_normalization();
  integral *= nu->stat().max_normalization();
  integral *= rho->stat().max_normalization();
  integral *= sigma->stat().max_normalization();
  
  return integral;

} // NodesMaxIntegral_()

double MultiTreeFock::NodesMinIntegral_(FockTree* mu, FockTree* nu, 
                                        FockTree* rho, FockTree* sigma) {
   
  double integral;
  
  double rho_sigma_max_dist = rho->bound().MaxDistanceSq(sigma->bound());
  
  double mu_nu_max_dist = mu->bound().MaxDistanceSq(nu->bound());
  
  DHrectBound<2> mu_nu_ave;
  mu_nu_ave.WeightedAverageBoxesInit(mu->stat().min_bandwidth(), mu->stat().max_bandwidth(), 
                                     mu->bound(), nu->stat().min_bandwidth(), 
                                     nu->stat().max_bandwidth(), nu->bound());
  DHrectBound<2> rho_sigma_ave;
  rho_sigma_ave.WeightedAverageBoxesInit(rho->stat().min_bandwidth(), rho->stat().max_bandwidth(), 
                                         rho->bound(), sigma->stat().min_bandwidth(), 
                                         sigma->stat().max_bandwidth(), sigma->bound());
                                         
  double four_way_max_dist = mu_nu_ave.MaxDistanceSq(rho_sigma_ave);
  
  /*
  double mu_min_band = mu->stat().min_bandwidth();
  double nu_min_band = nu->stat().min_bandwidth();
  double rho_min_band = rho->stat().min_bandwidth();
  double sigma_min_band = sigma->stat().min_bandwidth();
  */
  
  double mu_max_band = mu->stat().max_bandwidth();
  double nu_max_band = nu->stat().max_bandwidth();
  double rho_max_band = rho->stat().max_bandwidth();
  double sigma_max_band = sigma->stat().max_bandwidth();  
  
  // Make this as efficient as possible
  
  integral = eri::DistanceIntegral(mu_max_band, nu_max_band, 
                                   rho_max_band, sigma_max_band, 
                                   mu_nu_max_dist, rho_sigma_max_dist, 
                                   four_way_max_dist);
                                 
  
  //printf("min_integral: %g\n", integral);
                                   
  //integral *= NodeMinNorm_(mu) * NodeMinNorm_(nu) * 
  //  NodeMinNorm_(rho) * NodeMinNorm_(sigma);

  integral *= mu->stat().min_normalization();
  integral *= nu->stat().min_normalization();
  integral *= rho->stat().min_normalization();
  integral *= sigma->stat().min_normalization();
  
  
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
                                      
  integral *= NodeAveNorm_(mu) * NodeAveNorm_(nu) * 
    NodeAveNorm_(rho) * NodeAveNorm_(sigma);
  
  return integral;

} // NodesMidpointIntegral_()


bool MultiTreeFock::RectangleOnDiagonal_(FockTree* mu, 
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


index_t MultiTreeFock::CountOnDiagonal_(SquareTree* rho_sigma) {
  
  index_t on_diagonal;
  
  // one of these should be square, which one?
  SquareTree* left_child = rho_sigma->left();
  SquareTree* right_child = rho_sigma->right();
  
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

void MultiTreeFock::DensityFactor_(double* up_bound, double* low_bound, 
                                   double density_upper, 
                                   double density_lower) {
  
  // Need to account for the change in bounds if the density matrix entries 
  // are negative.  If the density lower bound is negative, then the lower 
  // bound becomes the largest integral times the density lower bound, 
  // instead of the smallest integral and vice versa.  
  
  double old_up_bound = *up_bound;
  if (density_upper >= 0.0) {
    *up_bound = *up_bound * density_upper;
  }
  else {
    *up_bound = *low_bound * density_upper;
  }
  if (density_lower >= 0.0) {
    *low_bound = *low_bound * density_lower;
  }
  else {
    *low_bound = old_up_bound * density_lower;
  }
  
  DEBUG_ASSERT(*up_bound >= *low_bound);
  
  
}

void MultiTreeFock::CountFactorCoulomb_(double* up_bound, double* low_bound, 
                                        double* approx_val, double* allowed_error,
                                        FockTree* rho, FockTree* sigma) {

  // Need to make this work with on diagonal non-square boxes
  if (RectangleOnDiagonal_(rho, sigma)) {
    index_t on_diagonal = CountOnDiagonal_(rho_sigma);
    index_t off_diagonal = (rho->count() * sigma->count()) - on_diagonal;
    *up_bound = (on_diagonal * *up_bound) + (2 * off_diagonal * *up_bound);
    // These two were missing, I think they needed to be here
    *low_bound = (on_diagonal * *low_bound) + (2 * off_diagonal * *low_bound);
    *approx_val = (on_diagonal * *approx_val) + (2 * off_diagonal * *approx_val);
    *allowed_error = (on_diagonal * *allowed_error)
      + (2 * off_diagonal * *allowed_error);
  }
  else if (rho != sigma) {
    *up_bound = 2 * *up_bound * rho->count() * sigma->count();
    *low_bound = 2 * *low_bound * rho->count() * sigma->count();
    *approx_val = 2 * *approx_val * rho->count() * sigma->count();
    *my_allowed_error = 2 * *my_allowed_error * rho->count() * sigma->count();
  }
  else {
    *up_bound = *up_bound * rho->count() * sigma->count();
    *low_bound = *low_bound * rho->count() * sigma->count();
    *approx_val = *approx_val * rho->count() * sigma->count();
    *allowed_error = *allowed_error * rho->count() * sigma->count();
  }
  
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
  

} // CountFactorCoulomb_


void MultiTreeFock::SchwartzBound_(SquareTree* mu_nu, SquareTree* rho_sigma,
                                     double* upper, double* lower) {

  double up_bound = mu_nu->stat().max_schwartz_factor();
  up_bound *= rho_sigma->stat().max_schwartz_factor();
  
  double low_bound = 0.0;
  
  // densities
  DensityFactor_(&up_bound, &low_bound, rho_sigma->stat().density_upper_bound(),
                 rho_sigma->stat().density_lower_bound());
                 
  *upper = up_bound;
  *lower = low_bound;
  
} // SchwartzBound_

bool MultiTreeFock::CanPrune_(double upper, double lower, SquareTree* mu_nu, 
                              SquareTree* rho_sigma) {

  

}


bool MultiTreeFock::CanApproximateCoulomb_(SquareTree* mu_nu, 
                                           SquareTree* rho_sigma, 
                                           double* approx_out) {

  double schwartz_upper;
  double schwartz_lower;
  SchwartzBound_(mu_nu, rho_sigma, &schwartz_upper, &schwartz_lower);

  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  double my_allowed_error = mu_nu->stat().remaining_epsilon() 
    / mu_nu->stat().remaining_references();
  
  if (CanPrune_(schwartz_upper, schwartz_lower, mu_nu, rho_sigma)) {
  
  }
  else {
  
  }
  
  double up_bound = NodesMaxIntegral_(mu_nu, rho_sigma);
  if (fabs(up_bound) < bounds_cutoff_) {
    up_bound = 0.0;
  } 
                    
  double low_bound = NodesMinIntegral_(mu_nu, rho_sigma);
  if (fabs(low_bound) < bounds_cutoff_) {
    low_bound = 0.0;
  } 
  
  //printf("up_bound: %g, low_bound: %g\n", up_bound, low_bound);
  DEBUG_ASSERT(up_bound >= low_bound);
  
  
  
  DensityFactor_(&up_bound, &low_bound, rho_sigma->stat().density_upper_bound(),
                 rho_sigma->stat().density_lower_bound());
      
  //double approx_val = NodesMidpointIntegral_(mu, nu, rho, sigma);
  double approx_val = 0.5 * (up_bound + low_bound);
  
  //double lost_error = max((up_bound - approx_val), (approx_val - low_bound));
  
  //printf("up: %g, low: %g, approx: %g\n", up_bound, low_bound, approx_val);
  
  // Multiply by number of references here, make sure to account for symmetry
  
  CountFactorCoulomb_(&up_bound, &low_bound, &approx_val, &my_allowed_error, 
                      rho, sigma);
  
  double my_max_error = max((up_bound - approx_val), 
                            (approx_val - low_bound));
                            
  double lost_error = my_max_error;
  
  // if using absolute error, then existing code is fine
  
  //bool below_cutoff;
  
  // For hybrid error 
  // assuming epsilon coulomb is the relative error tolerance
  // took out the <=, just in case
  // I think this is wrong, since the upper and lower bounds are initialized 
  // at 0.0, this means it always prunes with absolute error
  /*
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
  */
  
  if (relative_error_) {
  
    //my_max_error *= mu_nu->stat().entry_lower_bound();
    my_allowed_error *= mu_nu->stat().entry_lower_bound();
    
  }
  
  DEBUG_ONLY(*approx_out = BIG_BAD_NUMBER);
  
  //printf("max_err: %g, allowed_err: %g\n", my_max_error, my_allowed_error);
  
  if (my_max_error <= my_allowed_error) {
    
    /*
    if (my_max_error > 0.0) {
      printf("real prune\n");
    }
    */
    
    can_prune = true;
    
    /*
    printf("up_bound: %g, low_bound: %g, approx: %g\n", up_bound, low_bound, 
           approx_val);
    
    printf("max_err: %g, allowed_err: %g, eps: %g\n", my_max_error, 
           my_allowed_error, mu_nu->stat().remaining_epsilon());
    */
    
    *approx_out = approx_val;
    
    double new_entry_lower_bound = mu_nu->stat().entry_lower_bound() + approx_val;

    if (relative_error_ && (new_entry_lower_bound != 0.0)) {
      lost_error /= new_entry_lower_bound;
    }
    else if (relative_error_) {
      lost_error = 0.0;
    }
    
    //mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon()
    //                                    - lost_error);
      
    
    DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
    
    /*
    if (below_cutoff) {
      num_absolute_prunes_++;
    }
    else {
      num_relative_prunes_++;
    } 
    */   
  }
  
  //printf("can_prune: %d\n", can_prune);
  
  return can_prune;

}


bool MultiTreeFock::CanApproximateExchange_(SquareTree* mu_nu, 
                                            SquareTree* rho_sigma, 
                                            double* approximate_value) { 
                                            
  FockTree* mu = mu_nu->query1();
  FockTree* nu = mu_nu->query2();
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  // If it is possible to prune here, that will still be true for the two 
  // children, where we won't have to try to handle the near symmetry
  if (RectangleOnDiagonal_(rho, sigma)) {
    DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
    return false;
  }
  
  
  bool can_prune = false;
  
  double up_bound = NodesMaxIntegral_(mu, rho, nu, sigma);
  if (fabs(up_bound) < bounds_cutoff_) {
    up_bound = 0.0;
  } 
  
  double low_bound = NodesMinIntegral_(mu, rho, nu, sigma);
  if (fabs(low_bound) < bounds_cutoff_) {
    low_bound = 0.0;
  } 
  
  Vector mu_center;
  mu->bound().CalculateMidpoint(&mu_center);
  Vector nu_center;
  nu->bound().CalculateMidpoint(&nu_center);
  Vector rho_center;
  rho->bound().CalculateMidpoint(&rho_center);
  Vector sigma_center;
  sigma->bound().CalculateMidpoint(&sigma_center);
  
  //double approx_val = NodesMidpointIntegral_(mu, rho, nu, sigma);
  double approx_val = 0.5 * (up_bound + low_bound);
  
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
    
  if (rho != sigma) {
    
    /*
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
    */
    
    double mu_sigma_upper = NodesMaxIntegral_(mu, sigma, nu, rho);
    if (fabs(mu_sigma_upper) < bounds_cutoff_) {
      mu_sigma_upper = 0.0;
    } 
    
    double mu_sigma_lower = NodesMinIntegral_(mu, sigma, nu, rho);
    if (fabs(mu_sigma_lower) < bounds_cutoff_) {
      mu_sigma_lower = 0.0;
    }     

    //double mu_sigma_ave = NodesMidpointIntegral_(mu, sigma, nu, rho);
    
    up_bound += mu_sigma_upper;
    
    low_bound += mu_sigma_lower;
    
    approx_val += 0.5 * (mu_sigma_upper + mu_sigma_lower);
    
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    
  }
  
  // Because the exchange integrals have an extra 1/2 factor
  up_bound = up_bound * 0.5;
  low_bound = low_bound * 0.5;
  approx_val = approx_val * 0.5;
  
  //double lost_error = max((up_bound - approx_val), (approx_val - low_bound));
  
  up_bound *= rho->count() * sigma->count();
  low_bound *= rho->count() * sigma->count();
  approx_val *= rho->count() * sigma->count();
  
  // what about counting the references twice
  
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
  /*
  double my_allowed_error = epsilon_exchange_ * rho->count() * sigma->count()
    / (number_of_basis_functions_ * number_of_basis_functions_);
  */
  
  double my_allowed_error = mu_nu->stat().remaining_epsilon() * rho->count() 
    * sigma->count() / mu_nu->stat().remaining_references();
  
  // The total error I'm incurring is the max error for one integral times 
  // the number of approximations I'm making
  DEBUG_ASSERT(up_bound >= approx_val);
  DEBUG_ASSERT(approx_val >= low_bound);
  double my_max_error = max((up_bound - approx_val), 
                            (approx_val - low_bound));
                            
  double lost_error = my_max_error;
  
  if (relative_error_) {
    my_allowed_error *= mu_nu->stat().entry_lower_bound();
  }
  
  DEBUG_ONLY(*approximate_value = BIG_BAD_NUMBER);
  if (my_max_error <= my_allowed_error) {
    
    /*
    printf("up_bound: %g, low_bound: %g, approx: %g\n", up_bound, low_bound, 
           approx_val);
    
    printf("max_err: %g, allowed_err: %g, eps: %g\n", my_max_error, 
           my_allowed_error, mu_nu->stat().remaining_epsilon());
    */
    
    can_prune = true;
    
    *approximate_value = approx_val;
    //printf("approx_val = %g\n", *approximate_value);
    

    //double lost_error = my_max_error;
    double new_entry_lower_bound = mu_nu->stat().entry_lower_bound() + approx_val;
    
    if (relative_error_ && (new_entry_lower_bound != 0.0)) {
      lost_error /= new_entry_lower_bound;
    }
    else if (relative_error_) {
      lost_error = 0.0;
    }
    
    mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon()
                                        - lost_error);
    
    
    DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
    
  }
  
  return can_prune;
  
} // CanApproximateExchange_



void MultiTreeFock::ComputeCoulombBaseCase_(SquareTree* mu_nu, 
                                            SquareTree* rho_sigma) {
  
  FockTree* mu = mu_nu->query1();
  FockTree* nu = mu_nu->query2();
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  double max_entry = -DBL_INF;
  double min_entry = DBL_INF;
  
  DEBUG_ASSERT(mu->end() > nu->begin());
  DEBUG_ASSERT(rho->end() > sigma->begin());
  
  for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
    
    double alpha_mu = exponents_[mu_index];
    double mu_norm = eri::ComputeNormalization(alpha_mu, momenta_[mu_index]);
    
    for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
      
      double alpha_nu = exponents_[nu_index];
      double nu_norm = eri::ComputeNormalization(alpha_nu, momenta_[nu_index]);
      
      
      double integral_value = coulomb_matrix_.ref(mu_index, nu_index);
      
      for (index_t rho_index = rho->begin(); rho_index < rho->end();
           rho_index++) {
        
        double alpha_rho = exponents_[rho_index];
        double rho_norm = eri::ComputeNormalization(alpha_rho, 
                                                    momenta_[rho_index]);
        
        for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
             sigma_index++) {
          
          Vector mu_vec;
          centers_.MakeColumnVector(mu_index, &mu_vec);
          Vector nu_vec;
          centers_.MakeColumnVector(nu_index, &nu_vec);
          Vector rho_vec;
          centers_.MakeColumnVector(rho_index, &rho_vec);
          Vector sigma_vec;
          centers_.MakeColumnVector(sigma_index, &sigma_vec);
          
          double alpha_sigma = exponents_[sigma_index];
          
          double sigma_norm = eri::ComputeNormalization(alpha_sigma, 
                                                        momenta_[sigma_index]);
          
          // Multiply by normalization to the fourth, since it appears 
          // once in each of the four integrals
          
          double this_integral = density_matrix_.ref(rho_index, sigma_index) * 
            mu_norm * nu_norm * rho_norm * sigma_norm * 
            eri::SSSSIntegral(alpha_mu, mu_vec, alpha_nu, nu_vec, alpha_rho, 
                              rho_vec, alpha_sigma, sigma_vec);
          
          // this line gets it right
          // double this_integral = ComputeSingleIntegral_(mu_vec, nu_vec, rho_vec, sigma_vec);
          if (rho != sigma) {
            
            this_integral = this_integral * 2;
          }
          
          integral_value = integral_value + this_integral;
          
        } // sigma
        
      } // rho
      
      // Set both to account for mu/nu symmetry
      coulomb_matrix_.set(mu_index, nu_index, integral_value);
      // Necessary to fill in the lower triangle
      if (mu != nu) {
        coulomb_matrix_.set(nu_index, mu_index, integral_value);
      }
      
      if (integral_value > max_entry) {
        max_entry = integral_value;
      }
      if (integral_value < min_entry) {
        min_entry = integral_value;
      }
      
    } // nu
    
  } // mu
  
  mu_nu->stat().set_entry_upper_bound(max_entry);
  mu_nu->stat().set_entry_lower_bound(min_entry);
  
  index_t new_refs = rho->count() * sigma->count();
  
  if (rho != sigma) {
    new_refs = 2 * new_refs;
    DEBUG_ASSERT(!((rho->begin() < sigma->begin()) && 
                   (rho->end() > sigma->end())));
    DEBUG_ASSERT(!((rho->begin() > sigma->begin()) && 
                   (rho->end() < sigma->end())));
  }
  
  //mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
  //                                       - new_refs);
  
} // ComputeCoulombBaseCase_



void MultiTreeFock::ComputeExchangeBaseCase_(SquareTree* mu_nu, 
                                             SquareTree* rho_sigma) {
  
  FockTree* mu = mu_nu->query1();
  FockTree* nu = mu_nu->query2();
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  double max_entry = -DBL_INF;
  double min_entry = DBL_INF;
  
  DEBUG_ASSERT(mu->end() > nu->begin());
  DEBUG_ASSERT(rho->end() > sigma->begin());
  
  for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
    
    double alpha_mu = exponents_[mu_index];
    double mu_norm = eri::ComputeNormalization(alpha_mu, momenta_[mu_index]);
    
    for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
      
      double alpha_nu = exponents_[nu_index];
      double nu_norm = eri::ComputeNormalization(alpha_nu, momenta_[nu_index]);
      
      double integral_value = exchange_matrix_.ref(mu_index, nu_index);
      
      for (index_t rho_index = rho->begin(); rho_index < rho->end();
           rho_index++) {
        
        double alpha_rho = exponents_[rho_index];
        double rho_norm = eri::ComputeNormalization(alpha_rho, 
                                                    momenta_[rho_index]);
        
        for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
             sigma_index++) {
          
          
          Vector mu_vec;
          centers_.MakeColumnVector(mu_index, &mu_vec);
          Vector nu_vec;
          centers_.MakeColumnVector(nu_index, &nu_vec);
          Vector rho_vec;
          centers_.MakeColumnVector(rho_index, &rho_vec);
          Vector sigma_vec;
          centers_.MakeColumnVector(sigma_index, &sigma_vec);
          
          double alpha_sigma = exponents_[sigma_index];
          
          double sigma_norm = eri::ComputeNormalization(alpha_sigma, 
                                                        momenta_[sigma_index]);          
                    
          // multiply by 0.5 for exchange matrix
          double kl_integral = 0.5 * density_matrix_.ref(rho_index, sigma_index) * 
            mu_norm * nu_norm * rho_norm * sigma_norm * 
            eri::SSSSIntegral(alpha_mu, mu_vec, alpha_rho, rho_vec, 
                              alpha_nu, nu_vec, alpha_sigma, sigma_vec);
          
          integral_value = integral_value + kl_integral;
          
          // Account for the rho-sigma partial symmetry
          // No need to make this rectangle safe - base cases are square on 
          // diagonal
          if (rho != sigma) {  
            
            DEBUG_ASSERT(density_matrix_.ref(rho_index, sigma_index) == 
                         density_matrix_.ref(sigma_index, rho_index));
            
            double lk_integral = density_matrix_.ref(sigma_index, rho_index) * 
              mu_norm * nu_norm * rho_norm * sigma_norm * 
              eri::SSSSIntegral(alpha_mu, mu_vec, alpha_sigma, sigma_vec, 
                                alpha_nu, nu_vec, alpha_rho, rho_vec) * 0.5;
            
            integral_value = integral_value + lk_integral;
            
          }
        } // sigma
        
      } // rho
      
      // Set both to account for mu/nu symmetry
      exchange_matrix_.set(mu_index, nu_index, integral_value);
      // Necessary to fill in the lower triangle
      if (mu != nu) {
        exchange_matrix_.set(nu_index, mu_index, integral_value);
      }
      
      if (integral_value > max_entry) {
        max_entry = integral_value;
      }
      if (integral_value < min_entry) {
        min_entry = integral_value;
      }        
      
    } // nu
    
  } // mu
  
  mu_nu->stat().set_entry_upper_bound(max_entry);
  mu_nu->stat().set_entry_lower_bound(min_entry);
  
  index_t new_refs = rho->count() * sigma->count();
  if (rho != sigma) {
    new_refs = 2 * new_refs;
  }
  mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                         - new_refs);
  
} // ComputeExchangeBaseCase_


void MultiTreeFock::FillApproximationCoulomb_(SquareTree* mu_nu, 
                                              SquareTree* rho_sigma,
                                              double integral_approximation) {
  
  FockTree* mu = mu_nu->query1();
  FockTree* nu = mu_nu->query2();
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  // if mu and nu overlap, recursively call this on children
  // else if rho and sigma overlap, recursively call on children
  // Make sure to update the integral approximation (for both) 
  // and the bounds if splitting mu_nu
  // else do the below
  
  if (RectangleOnDiagonal_(mu, nu)) {
    
    PropagateBoundsDown_(mu_nu);
    
    // this doesn't hold because I haven't propagated them down this far
    DEBUG_ASSERT(mu_nu->stat().remaining_references() ==
                 mu_nu->left()->stat().remaining_references());
    
    FillApproximationCoulomb_(mu_nu->left(), rho_sigma, 
                              integral_approximation);
    FillApproximationCoulomb_(mu_nu->right(), rho_sigma, 
                              integral_approximation);
    
    PropagateBoundsUp_(mu_nu);
    
  }
  else if (RectangleOnDiagonal_(rho, sigma)) {
    
    FillApproximationCoulomb_(mu_nu, rho_sigma->left(), 
                              integral_approximation);
    FillApproximationCoulomb_(mu_nu, rho_sigma->right(), 
                              integral_approximation);
    
    // Because the approximation has been counted twice
    mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() - 
                                        integral_approximation);
    mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() - 
                                        integral_approximation);
    
  }
  else {
    
    
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
      
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
        
        double new_value = coulomb_matrix_.ref(mu_index, nu_index) + 
        integral_approximation;
        // Set both to account for mu/nu symmetry
        // This uses the same logic as the base case
        coulomb_matrix_.set(mu_index, nu_index, new_value);
        if (mu != nu) {
          coulomb_matrix_.set(nu_index, mu_index, new_value);
        }
        
      } // nu
      
    } // mu
    
    mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() + 
                                        integral_approximation);
    
    mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() + 
                                        integral_approximation);
    
    index_t new_refs = rho->count() * sigma->count();
    
    if (rho != sigma) {
      DEBUG_ASSERT(!((rho->begin() <= sigma->begin()) && 
                     (rho->end() >= sigma->end())));
      DEBUG_ASSERT(!((rho->begin() >= sigma->begin()) && 
                     (rho->end() <= sigma->end())));
      new_refs = 2 * new_refs;
    }
    
    // reduces remaining references, but never alters epsilon
    //mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() - new_refs);
    
    
    mu_nu->stat().set_approximation_val(integral_approximation);
    
  }
  
} // FillApproximationCoulomb_()


void MultiTreeFock::FillApproximationExchange_(SquareTree* mu_nu, 
                                               SquareTree* rho_sigma,
                                               double integral_approximation) {
  
  FockTree* mu = mu_nu->query1();
  FockTree* nu = mu_nu->query2();
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  if (RectangleOnDiagonal_(mu, nu)) {
    
    PropagateBoundsDown_(mu_nu);
    
    DEBUG_ASSERT(mu_nu->stat().remaining_references() ==
                 mu_nu->left()->stat().remaining_references());
    
    FillApproximationExchange_(mu_nu->left(), rho_sigma, 
                               integral_approximation);
    FillApproximationExchange_(mu_nu->right(), rho_sigma, 
                               integral_approximation);
    
    PropagateBoundsUp_(mu_nu);
    
  }
  else if (RectangleOnDiagonal_(rho, sigma)) {
    
    FillApproximationExchange_(mu_nu, rho_sigma->left(), 
                               integral_approximation);
    FillApproximationExchange_(mu_nu, rho_sigma->right(), 
                               integral_approximation);
    
    // Because the approximation has been counted twice
    mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() - 
                                        integral_approximation);
    mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() - 
                                        integral_approximation);
    
  }
  else {
    
    for (index_t mu_index = mu->begin(); mu_index < mu->end(); mu_index++) {
      
      for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
        
        double new_value = exchange_matrix_.ref(mu_index, nu_index) + 
        integral_approximation;
        // Set both to account for mu/nu symmetry
        // This uses the same logic as the base case
        exchange_matrix_.set(mu_index, nu_index, new_value);
        // I could probably check for this on the outside and make this function
        // faster
        if (mu != nu) {
          exchange_matrix_.set(nu_index, mu_index, new_value);
        }
        
      } // nu
      
    } // mu
    
    mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() + 
                                        integral_approximation);
    
    mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() + 
                                        integral_approximation);
    
    index_t new_refs = rho->count() * sigma->count();
    if (rho != sigma) {
      new_refs = 2 * new_refs;
    }
    mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                           - new_refs);
    
    mu_nu->stat().set_approximation_val(integral_approximation);
    
  }
  
} // FillApproximationExchange_()


void MultiTreeFock::PropagateBoundsDown_(SquareTree* query) {
  
  query->left()->stat().set_remaining_references(query->stat().remaining_references());
  query->right()->stat().set_remaining_references(query->stat().remaining_references());
  
  query->left()->stat().set_remaining_epsilon(query->stat().remaining_epsilon());
  query->right()->stat().set_remaining_epsilon(query->stat().remaining_epsilon());
  
  if (query->stat().approximation_val() != 0.0) {
    
    query->left()->stat().set_entry_upper_bound(query->left()->stat().entry_upper_bound() + 
                                                query->stat().approximation_val());
    query->right()->stat().set_entry_upper_bound(query->right()->stat().entry_upper_bound() + 
                                                 query->stat().approximation_val());
    
    query->left()->stat().set_entry_lower_bound(query->left()->stat().entry_lower_bound() + 
                                                query->stat().approximation_val());
    query->right()->stat().set_entry_lower_bound(query->right()->stat().entry_lower_bound() + 
                                                 query->stat().approximation_val());
    
    query->stat().set_approximation_val(0.0);
    
  }
  
} // PropagateBoundsDown_()

void MultiTreeFock::PropagateBoundsUp_(SquareTree* query) {
  
  double min_entry = query->left()->stat().entry_lower_bound();
  double max_entry = query->left()->stat().entry_upper_bound();
  
  min_entry = min(min_entry, 
                  query->right()->stat().entry_lower_bound());
  max_entry = max(max_entry, 
                  query->right()->stat().entry_upper_bound());
  
  query->stat().set_entry_upper_bound(max_entry);
  query->stat().set_entry_lower_bound(min_entry);
  
  // Do I actually need to set this here?
  query->stat().set_remaining_references(query->left()->stat().remaining_references());
  
  DEBUG_ASSERT(query->stat().remaining_references() == 
               query->right()->stat().remaining_references());
  
  // should I mess with the remaining epsilon in this function?
  
} // PropagateBoundsUp_()


void MultiTreeFock::ComputeCoulombRecursion_(SquareTree* query, 
                                             SquareTree* reference) {
  
  DEBUG_ASSERT(query->query1()->end() > query->query2()->begin());
  DEBUG_ASSERT(reference->query1()->end() > reference->query2()->begin());
  
  double integral_approximation;
  
  if (query->is_leaf() && reference->is_leaf()) {
    
    coulomb_base_cases_++;
    ComputeCoulombBaseCase_(query, reference);
    
  }
  else if(CanApproximateCoulomb_(query, reference, &integral_approximation)) {
    
    DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
    
    coulomb_approximations_++;
    
    FillApproximationCoulomb_(query, reference, integral_approximation);
    
  }        
  else if (query->is_leaf()) {
    // Don't need to propagate stats
    
    ComputeCoulombRecursion_(query, reference->left());
    ComputeCoulombRecursion_(query, reference->right());
    
  }
  else if (reference->is_leaf()) {
    // Need to propagate some stats
    
    PropagateBoundsDown_(query);
    
    ComputeCoulombRecursion_(query->left(), reference);
    ComputeCoulombRecursion_(query->right(), reference);
    
    PropagateBoundsUp_(query);
    
  }
  else {
    
    PropagateBoundsDown_(query);
    
    ComputeCoulombRecursion_(query->left(), reference->left());
    ComputeCoulombRecursion_(query->left(), reference->right());
    
    ComputeCoulombRecursion_(query->right(), reference->left());
    ComputeCoulombRecursion_(query->right(), reference->right());
    
    PropagateBoundsUp_(query);
    
  }   
  
} // ComputeCoulombRecursion_()


void MultiTreeFock::ComputeExchangeRecursion_(SquareTree* query, 
                                              SquareTree* reference) {
  
  double integral_approximation;
  
  if (query->is_leaf() && reference->is_leaf()) {
    
    exchange_base_cases_++;
    ComputeExchangeBaseCase_(query, reference);
    
  }
  else if (CanApproximateExchange_(query, reference, &integral_approximation)) {
    
    DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
    
    exchange_approximations_++;
    
    FillApproximationExchange_(query, reference, integral_approximation);
    
  }        
  else if (query->is_leaf()) {
    // Don't need to propagate stats
    
    ComputeExchangeRecursion_(query, reference->left());
    ComputeExchangeRecursion_(query, reference->right());
    
  }
  else if (reference->is_leaf()) {
    // Need to propagate some stats
    
    PropagateBoundsDown_(query);
    
    ComputeExchangeRecursion_(query->left(), reference);
    ComputeExchangeRecursion_(query->right(), reference);
    
    PropagateBoundsUp_(query);
    
  }
  else {
    
    PropagateBoundsDown_(query);
    
    ComputeExchangeRecursion_(query->left(), reference->left());
    ComputeExchangeRecursion_(query->left(), reference->right());
    
    ComputeExchangeRecursion_(query->right(), reference->left());
    ComputeExchangeRecursion_(query->right(), reference->right());
    
    PropagateBoundsUp_(query);
    
  }
  
} // ComputeExchangeRecursion_()

// NOTE: not sure if the max bandwidth really corresponds to the max integral
void MultiTreeFock::SetEntryBounds_() {
  
  double density_upper = square_tree_->stat().density_upper_bound();
  double density_lower = square_tree_->stat().density_lower_bound();
  
  double exp_upper = tree_->stat().max_bandwidth();
  double exp_lower = tree_->stat().min_bandwidth();
  
  DEBUG_ASSERT(density_upper >= density_lower);
  DEBUG_ASSERT(exp_upper >= exp_lower);
  
  double entry_upper;
  double entry_lower;
  
  double max_dist = 
      square_tree_->query1()->bound().MaxDistanceSq(square_tree_->query2()->bound());
      
  double max_norm = eri::ComputeNormalization(exp_upper, 0);
  double min_norm = eri::ComputeNormalization(exp_lower, 0);
      
  // this may not be normalized
  double max_integral = eri::DistanceIntegral(exp_lower, exp_lower, exp_lower, 
                                              exp_lower, 0.0, 0.0, 0.0);
  max_integral *= pow(max_norm, 4.0);
                                              
  double min_integral = eri::DistanceIntegral(exp_upper, exp_upper, exp_upper, 
                                              exp_upper, max_dist, 
                                              max_dist, max_dist);
  min_integral *= pow(min_norm, 4.0);
  
  if (density_upper > 0) {
    // then, the largest value is when all the distances are 0
    
    entry_upper = density_upper * number_of_basis_functions_ * 
        number_of_basis_functions_ * max_integral;
    
  }
  else {
    // then, the largest value is when all the distances are max
    
    entry_upper = density_upper * min_integral * number_of_basis_functions_ 
                 * number_of_basis_functions_;
    
  }
  
  if (density_lower > 0) {
    //then, the smallest value is when all the distances are max
    
    entry_lower = density_lower * min_integral * 
        number_of_basis_functions_ * number_of_basis_functions_;
    
  }
  else {
    
    entry_lower = density_lower * number_of_basis_functions_ * 
        number_of_basis_functions_ * max_integral;
    
  }
  
  DEBUG_ASSERT(entry_upper >= entry_lower);
  
  square_tree_->stat().set_entry_upper_bound(entry_upper);
  square_tree_->stat().set_entry_lower_bound(entry_lower);
  
} // SetEntryBounds_


void MultiTreeFock::ResetTreeForExchange_(SquareTree* root) {
  
  if (root != NULL) {
    
    root->stat().set_remaining_references(number_of_basis_functions_ * 
                                          number_of_basis_functions_);
                                          
    root->stat().set_remaining_epsilon((1.0 - epsilon_split_) * epsilon_);
    
    root->stat().set_approximation_val(0.0);
    
    ResetTreeForExchange_(root->left());
    ResetTreeForExchange_(root->right());  
    
  }
  
} // ResetTreeForExchange_()



void MultiTreeFock::ResetTree_(SquareTree* root) {
  
  double max_density;
  double min_density;
  
  /*
  double min_exp;
  double max_exp;
  */
  
  if (root->is_leaf()) {
    
    max_density = -DBL_INF;
    min_density = DBL_INF;
    
    
    /*min_exp = DBL_INF;
    max_exp = DBL_INF;
    */
    
    for (index_t i = root->query1()->begin(); i < root->query1()->end(); 
         i++) {
      
      for (index_t j = root->query2()->begin(); j < root->query2()->end(); 
           j++) {
        
        double this_density = density_matrix_.ref(i, j);
        if (this_density > max_density) {
          max_density = this_density;
        }
        if (this_density < min_density) {
          min_density = this_density;
        }
        
      } // j
      
      /*
      if (exponents_[i] < min_exp) {
        min_exp = exponents_[i];
      }
      if (exponents_[i] > max_exp) {
        max_exp = exponents_[i];
      }
      */
      
    } // i
    
    
  } // leaf
  else {
    
    ResetTree_(root->left());
    ResetTree_(root->right());
    
    max_density = max(root->left()->stat().density_upper_bound(), 
                      root->right()->stat().density_upper_bound());
    min_density = min(root->left()->stat().density_lower_bound(), 
                      root->right()->stat().density_lower_bound());
                      
    
    
  } // non-leaf
  
  root->stat().set_density_upper_bound(max_density);
  root->stat().set_density_lower_bound(min_density);
  
  root->stat().set_remaining_references(number_of_basis_functions_ * 
                                        number_of_basis_functions_);
  
  root->stat().set_approximation_val(0.0);
  
  /*
  root->query1().stat().set_max_bandwidth(max_exp);
  root->query1().stat().set_min_bandwidth(min_exp);
  */
  
} // ResetTree_()

void MultiTreeFock::SetExponentBounds_(FockTree* tree) {

  if (tree->is_leaf()) {
  
    double max_exp = -DBL_MAX;
    double min_exp = DBL_MAX;
  
    for (index_t i = tree->begin(); i < tree->end(); i++) {
    
      if (max_exp < exponents_[i]) {
        max_exp = exponents_[i];
      }
      
      if (min_exp > exponents_[i]) {
        min_exp = exponents_[i];
      }
    
    } // for i
    
    tree->stat().set_max_bandwidth(max_exp);
    tree->stat().set_min_bandwidth(min_exp);
    
    double max_norm = eri::ComputeNormalization(max_exp, 0);
    double min_norm = eri::ComputeNormalization(min_exp, 0);
    
    tree->stat().set_max_normalization(max_norm);
    tree->stat().set_min_normalization(min_norm);
  
  } // is leaf
  else {
  
    SetExponentBounds_(tree->left());
    SetExponentBounds_(tree->right());
  
    tree->stat().set_max_bandwidth(max(tree->left()->stat().max_bandwidth(), 
                                       tree->right()->stat().max_bandwidth()));

    tree->stat().set_min_bandwidth(min(tree->left()->stat().min_bandwidth(), 
                                       tree->right()->stat().min_bandwidth()));
                                       
    tree->stat().set_max_normalization(max(tree->left()->stat().max_normalization(), 
                                           tree->right()->stat().max_normalization()));

    tree->stat().set_min_normalization(min(tree->left()->stat().min_normalization(), 
                                           tree->right()->stat().min_normalization()));
  
  } // not leaf

} // SetExponentBounds_

void MultiTreeFock::ApplyPermutation(ArrayList<index_t>& old_from_new, 
                                     Matrix* mat) {

  DEBUG_ASSERT(old_from_new.size() == mat->n_cols());
  
  Matrix temp_mat;
  temp_mat.Init(mat->n_rows(), mat->n_cols());
  
  for (index_t i = 0; i < old_from_new.size(); i++) {
  
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



///////////////////// public functions ////////////////////////////////////

void MultiTreeFock::ComputeFockMatrix() {

  fx_timer_start(module_, "multi_time");

  printf("====Computing J====\n");
  fx_timer_start(module_, "coulomb_recursion");
  ComputeCoulombRecursion_(square_tree_, square_tree_);  
  fx_timer_stop(module_, "coulomb_recursion");

  // Will need to be followed by clearing the tree and computing the exchange 
  // matrix
  // I think this is the only resetting the tree will need
  // the density and exponent bounds are not set

  printf("====Computing K====\n");
  SetEntryBounds_();
  ResetTreeForExchange_(square_tree_);
  fx_timer_start(module_, "exchange_recursion");
  ComputeExchangeRecursion_(square_tree_, square_tree_);
  fx_timer_stop(module_, "exchange_recursion");
    
  la::SubOverwrite(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  
  fx_timer_stop(module_, "multi_time");

} // ComputeFockMatrix()

void MultiTreeFock::UpdateMatrices(const Matrix& new_density) {
  
  density_matrix_.CopyValues(new_density);
  // this won't be correct when I switch to higher momentum
  ApplyPermutation(old_from_new_centers_, &density_matrix_);
  
  //density_matrix_.PrintDebug();
  
  // Reset tree density bounds
  ResetTree_(square_tree_);
  
  SetEntryBounds_();
  
  coulomb_matrix_.SetZero();
  exchange_matrix_.SetZero();
  fock_matrix_.SetZero();
  
} // UpdateMatrices()


void MultiTreeFock::OutputFockMatrix(Matrix* fock_out, Matrix* coulomb_out, 
                                     Matrix* exchange_out, 
                                     ArrayList<index_t>* old_from_new) {
  
  //printf("number_of_approximations_ = %d\n", number_of_approximations_);
  //printf("number_of_base_cases_ = %d\n\n", number_of_base_cases_);
  //fx_format_result(module_, "bandwidth", "%g", bandwidth_);
  fx_result_double(module_, "epsilon_coulomb", epsilon_coulomb_);
  fx_result_double(module_, "epsilon_exchange", epsilon_exchange_);
  fx_result_int(module_, "coulomb_approximations", 
                   coulomb_approximations_);
  fx_result_int(module_, "exchange_approximations", 
                   exchange_approximations_);
  fx_result_int(module_, "coulomb_base_cases", 
                   coulomb_base_cases_);
  fx_result_int(module_, "exchange_base_cases", 
                   exchange_base_cases_);
  
  if (fock_out) {
    fock_out->Copy(fock_matrix_);
    UnApplyPermutation(old_from_new_centers_, fock_out);
  }
  if (coulomb_out) {
    coulomb_out->Copy(coulomb_matrix_);
    UnApplyPermutation(old_from_new_centers_, coulomb_out);
  }
  if (exchange_out) {
    exchange_out->Copy(exchange_matrix_);
    UnApplyPermutation(old_from_new_centers_, exchange_out);
  }
  
  if (old_from_new) {
    old_from_new->InitCopy(old_from_new_centers_);
  }
    
} // OutputFockMatrix()



