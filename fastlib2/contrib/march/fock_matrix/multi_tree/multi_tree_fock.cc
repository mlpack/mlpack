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
  
  //double mu_nu_dist = mu_nu->query1()->bound().MinDistanceSq(mu_nu->query2()->bound());
  //double rho_sigma_dist = rho_sigma->query1()->bound().MinDistanceSq(rho_sigma->query2()->bound());
  
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
  
  num_integrals_computed_++;

  return integral;

} // NodesMaxIntegral(SquareTree)

double MultiTreeFock::NodesMinIntegral_(SquareTree* mu_nu, SquareTree* rho_sigma) {
  
  double integral = 2 * pow_pi_2point5_;
  
  //double mu_nu_dist = mu_nu->query1()->bound().MaxDistanceSq(mu_nu->query2()->bound());
  //double rho_sigma_dist = rho_sigma->query1()->bound().MaxDistanceSq(rho_sigma->query2()->bound());

  
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
  
  num_integrals_computed_++;

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
  
  num_integrals_computed_++;

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
  
  num_integrals_computed_++;

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
  
  num_integrals_computed_++;

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
  else if (right_child->query1() == right_child->query2()) {
    DEBUG_ASSERT(right_child->query1() == right_child->query2());
    on_diagonal = right_child->query1()->count() * 
      right_child->query2()->count();
  }
  else {
    on_diagonal = 0;
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
  
  // does this work if the upper and lower bounds are less than zero?
  
  DEBUG_ASSERT(density_upper >= density_lower);
  DEBUG_ASSERT(*up_bound >= *low_bound);
  
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
                                        SquareTree* rho_sigma) {

  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  if (RectangleOnDiagonal_(rho, sigma)) {
    index_t on_diagonal = CountOnDiagonal_(rho_sigma);
    index_t off_diagonal = (rho->count() * sigma->count()) - on_diagonal;
    *up_bound = (on_diagonal * *up_bound) + (2 * off_diagonal * *up_bound);
    *low_bound = (on_diagonal * *low_bound) + (2 * off_diagonal * *low_bound);
    // These two were missing, I think they needed to be here
    *approx_val = (on_diagonal * *approx_val) + (2 * off_diagonal * *approx_val);
    *allowed_error = (on_diagonal * *allowed_error)
      + (2 * off_diagonal * *allowed_error);
  }
  else if (rho != sigma) {
    *up_bound = 2 * *up_bound * rho->count() * sigma->count();
    *low_bound = 2 * *low_bound * rho->count() * sigma->count();
    *approx_val = 2 * *approx_val * rho->count() * sigma->count();
    *allowed_error = 2 * *allowed_error * rho->count() * sigma->count();
  }
  else {
    *up_bound = *up_bound * rho->count() * sigma->count();
    *low_bound = *low_bound * rho->count() * sigma->count();
    *approx_val = *approx_val * rho->count() * sigma->count();
    *allowed_error = *allowed_error * rho->count() * sigma->count();
  }
  
  DEBUG_ASSERT(*up_bound >= *approx_val);
  DEBUG_ASSERT(*approx_val >= *low_bound);
  

} // CountFactorCoulomb_


void MultiTreeFock::CountFactorExchange_(double* up_bound, double* low_bound, 
                                         double* approx_val, double* allowed_error,
                                         SquareTree* rho_sigma) {
  
  // the lower triangle has already been accounted for in the computation of the
  // bounds
  
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  *up_bound = *up_bound * rho->count() * sigma->count();
  *low_bound = *low_bound * rho->count() * sigma->count();
  *approx_val = *approx_val * rho->count() * sigma->count();
  *allowed_error = *allowed_error * rho->count() * sigma->count();
  
  DEBUG_ASSERT(*up_bound >= *approx_val);
  DEBUG_ASSERT(*approx_val >= *low_bound);
  
} // CountFactorExchange_


void MultiTreeFock::SchwartzBound_(SquareTree* mu_nu, SquareTree* rho_sigma,
                                     double* upper, double* lower) {

  double up_bound = mu_nu->stat().max_schwartz_factor();
  up_bound *= rho_sigma->stat().max_schwartz_factor();
  
  // can I do any better?
  // is this still true in the higher momenta case?
  double low_bound = 0.0;
  
  // densities
  DensityFactor_(&up_bound, &low_bound, rho_sigma->stat().density_upper_bound(),
                 rho_sigma->stat().density_lower_bound());
                 
  *upper = up_bound;
  *lower = low_bound;
  DEBUG_ASSERT(*upper >= *lower);
  
} // SchwartzBound_


// should the bounds have been multiplied by the appropriate counts?
// maybe not, will have to do it several times
bool MultiTreeFock::CanPruneCoulomb_(double* upper, double* lower, 
                                     double* approx_val,
                                     SquareTree* mu_nu, SquareTree* rho_sigma) {

  DEBUG_ASSERT(*upper >= *approx_val);
  DEBUG_ASSERT(*approx_val >= *lower);
  
  // works for absolute error
  double my_allowed_error = mu_nu->stat().remaining_epsilon() 
                            / (double)mu_nu->stat().remaining_references();
  
  if (relative_error_) {
   
    // this isn't necessarily the lower bound, it's the smallest the entry
    // can be in absolute value
    // so, if the bounds place the entry in (-1, 1), then the available error
    // should be multiplied by zero
    my_allowed_error *= mu_nu->stat().relative_error_bound();
    
  }
  
  CountFactorCoulomb_(upper, lower, approx_val, &my_allowed_error,
                      rho_sigma);
  
  double max_error = max(fabs(*upper - *approx_val), fabs(*lower - *approx_val));
  
  //printf("max_error: %g, allowed_error: %g\n\n", max_error, my_allowed_error);
  return (max_error <= my_allowed_error);

}


bool MultiTreeFock::CanPruneExchange_(double* upper, double* lower, 
                                      double* approx_val,
                                      SquareTree* mu_nu, SquareTree* rho_sigma) {
  
  DEBUG_ASSERT(*upper >= *approx_val);
  DEBUG_ASSERT(*approx_val >= *lower);
  
  double allowed_err = mu_nu->stat().remaining_epsilon() 
                       / (double)mu_nu->stat().remaining_references();
  
  if (relative_error_) {
    
    allowed_err *= mu_nu->stat().relative_error_bound();
    
  }
  
  
  CountFactorExchange_(upper, lower, approx_val, &allowed_err, 
                       rho_sigma);
  
  double max_error = max(fabs(*upper - *approx_val), fabs(*lower - *approx_val));
  
  //printf("max_error: %g, allowed_error: %g\n\n", max_error, my_allowed_error);
  return (max_error <= allowed_err);
  
} // CanPruneExchange_



bool MultiTreeFock::CanApproximateCoulomb_(SquareTree* mu_nu, 
                                           SquareTree* rho_sigma, 
                                           double* approx_out, 
                                           double* lost_error_out) {

  double lost_error;
  
  // Schwartz pruning
  if (schwartz_pruning_) {
    double schwartz_upper;
    double schwartz_lower;
    // this includes the density matrix entries
    SchwartzBound_(mu_nu, rho_sigma, &schwartz_upper, &schwartz_lower);

    double schwartz_approx = 0.5 * (schwartz_lower + schwartz_upper);
    
    
    //FockTree* rho = rho_sigma->query1();
    //FockTree* sigma = rho_sigma->query2();
    
    if (CanPruneCoulomb_(&schwartz_upper, &schwartz_lower, &schwartz_approx, 
                         mu_nu, rho_sigma)) {
    
      lost_error = max(fabs(schwartz_upper - schwartz_approx), 
                       fabs(schwartz_approx - schwartz_lower));
      
      // fill in the approximations and return
      *approx_out = schwartz_approx;
      
      // I think that modifying lost error to account for relative error needs
      // to be moved to the fill approximation functions, since they know the 
      // new entry bounds 
      
      // need to redistribute error
      // this isn't what I need to redistribute error
      // I need the new relative_error_bound, which has to be computed
      //double new_entry_lower_bound = mu_nu->stat().entry_lower_bound() 
      //                               + schwartz_approx;
      
      //if (relative_error_ && (new_entry_lower_bound != 0.0)) {
      //  lost_error /= new_entry_lower_bound;
      //}
      // if the smallest the entry can be in absolute value is 0, then in 
      // order to prune, the difference between my bounds must have been 0
      // therefore, I didn't incur any error with this prune
      //else if (relative_error_) {
      //  lost_error = 0.0;
      //}
      
      *lost_error_out = lost_error;
      
      // moved this to FillApproximationCoulomb_()
      //mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon()
      //                                    - lost_error);
      
      //DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
      
      
      num_schwartz_prunes_++;
      return true;
      
    }
  } // schwartz pruning
  
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

  
  if (CanPruneCoulomb_(&up_bound, &low_bound, &approx_val, mu_nu, rho_sigma)) {
    
    //lost_error = max(fabs(up_bound - approx_val), fabs(approx_val - low_bound))
    //              / mu_nu->stat().remaining_references();
    
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);
    
    lost_error = max(fabs(up_bound - approx_val), fabs(approx_val - low_bound));
    
    // fill in approximation
    *approx_out = approx_val;
    
    /*
    double new_entry_lower_bound = mu_nu->stat().entry_lower_bound() + approx_val;
    
    if (relative_error_ && (new_entry_lower_bound != 0.0)) {
      lost_error /= new_entry_lower_bound;
    }
    else if (relative_error_) {
      lost_error = 0.0;
    }
    */
    
    *lost_error_out = lost_error;
    
    // moved this to FillApproximation
    //mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon()
    //                                    - lost_error);
    
    //DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
    
    return true;
    
  }

  return false;

} // CanApproximateCoulomb_()


bool MultiTreeFock::CanApproximateExchange_(SquareTree* mu_nu, 
                                            SquareTree* rho_sigma, 
                                            double* approximate_value,
                                            double* lost_error_out) { 
                                            
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
  
  DEBUG_ASSERT(up_bound >= low_bound);  
  
  if (rho != sigma) {
    
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
    
    //approx_val += 0.5 * (mu_sigma_upper + mu_sigma_lower);
    
    //DEBUG_ASSERT(up_bound >= approx_val);
    //DEBUG_ASSERT(approx_val >= low_bound);
    DEBUG_ASSERT(up_bound >= low_bound);
    
  }
  
  // Because the exchange integrals have an extra 1/2 factor
  up_bound = up_bound * 0.5;
  low_bound = low_bound * 0.5;
  //approx_val = approx_val * 0.5;
  
  //double lost_error = max((up_bound - approx_val), (approx_val - low_bound));
  
  //DEBUG_ASSERT(up_bound >= approx_val);
  //DEBUG_ASSERT(approx_val >= low_bound);
  DEBUG_ASSERT(up_bound >= low_bound);
  
  DensityFactor_(&up_bound, &low_bound, rho_sigma->stat().density_upper_bound(), 
                 rho_sigma->stat().density_lower_bound());
  
  DEBUG_ASSERT(up_bound >= low_bound);
  
  double approx_val = 0.5 * (up_bound + low_bound);
  
  if (CanPruneExchange_(&up_bound, &low_bound, &approx_val, mu_nu, rho_sigma)) {
  
    DEBUG_ASSERT(up_bound >= approx_val);
    DEBUG_ASSERT(approx_val >= low_bound);

    double lost_error = max(fabs(up_bound - approx_val), 
                            fabs(approx_val - low_bound));

    can_prune = true;
    
    *approximate_value = approx_val;

    //double new_entry_lower_bound = mu_nu->stat().entry_lower_bound() + approx_val;
    
    *lost_error_out = lost_error;
    
    //mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon()
    //                                    - lost_error);
    
    //DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
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
    
    Vector mu_vec;
    centers_.MakeColumnVector(mu_index, &mu_vec);
    
    for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
      
      double alpha_nu = exponents_[nu_index];
      double nu_norm = eri::ComputeNormalization(alpha_nu, momenta_[nu_index]);
      
      Vector nu_vec;
      centers_.MakeColumnVector(nu_index, &nu_vec);
      
      double integral_value = coulomb_matrix_.ref(mu_index, nu_index);
      
      for (index_t rho_index = rho->begin(); rho_index < rho->end();
           rho_index++) {
        
        double alpha_rho = exponents_[rho_index];
        double rho_norm = eri::ComputeNormalization(alpha_rho, 
                                                    momenta_[rho_index]);
        Vector rho_vec;
        centers_.MakeColumnVector(rho_index, &rho_vec);
        
        for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
             sigma_index++) {
          
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
          
          num_integrals_computed_++;

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
  mu_nu->stat().set_relative_error_bound(min_entry, max_entry);
  
  index_t new_refs = rho->count() * sigma->count();
  
  // a base case can't be on the diagonal, so this is okay
  if (rho != sigma) {
    new_refs = 2 * new_refs;
    DEBUG_ASSERT(!((rho->begin() < sigma->begin()) && 
                   (rho->end() > sigma->end())));
    DEBUG_ASSERT(!((rho->begin() > sigma->begin()) && 
                   (rho->end() < sigma->end())));
  }
  
  mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                         - new_refs);
  
  
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
    
    Vector mu_vec;
    centers_.MakeColumnVector(mu_index, &mu_vec);
    
    for (index_t nu_index = nu->begin(); nu_index < nu->end(); nu_index++) {
      
      double alpha_nu = exponents_[nu_index];
      double nu_norm = eri::ComputeNormalization(alpha_nu, momenta_[nu_index]);
      
      Vector nu_vec;
      centers_.MakeColumnVector(nu_index, &nu_vec);
      
      double integral_value = exchange_matrix_.ref(mu_index, nu_index);
      
      for (index_t rho_index = rho->begin(); rho_index < rho->end();
           rho_index++) {
        
        double alpha_rho = exponents_[rho_index];
        double rho_norm = eri::ComputeNormalization(alpha_rho, 
                                                    momenta_[rho_index]);
        
        Vector rho_vec;
        centers_.MakeColumnVector(rho_index, &rho_vec);
        
        for (index_t sigma_index = sigma->begin(); sigma_index < sigma->end(); 
             sigma_index++) {
          
          
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
          
          num_integrals_computed_++;

          integral_value = integral_value + kl_integral;
          
          // Account for the rho-sigma partial symmetry
          // No need to make this rectangle safe - base cases are square on 
          // diagonal
          if (rho != sigma) {  
            
            // not actually true, since it's being read from a file
            //DEBUG_ASSERT(density_matrix_.ref(rho_index, sigma_index) == 
            //             density_matrix_.ref(sigma_index, rho_index));
            
            //DEBUG_APPROX_DOUBLE(density_matrix_.ref(rho_index, sigma_index),
            //                    density_matrix_.ref(sigma_index, rho_index),
            //                    1e-6);
            
            double lk_integral = density_matrix_.ref(sigma_index, rho_index) * 
              mu_norm * nu_norm * rho_norm * sigma_norm * 
              eri::SSSSIntegral(alpha_mu, mu_vec, alpha_sigma, sigma_vec, 
                                alpha_nu, nu_vec, alpha_rho, rho_vec) * 0.5;
            
            num_integrals_computed_++;

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
  mu_nu->stat().set_relative_error_bound(min_entry, max_entry);
  
  index_t new_refs = rho->count() * sigma->count();
  if (rho != sigma) {
    new_refs = 2 * new_refs;
  }
  mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                         - new_refs);
  
} // ComputeExchangeBaseCase_


void MultiTreeFock::FillApproximationCoulomb_(SquareTree* mu_nu, 
                                              SquareTree* rho_sigma,
                                              double integral_approximation,
                                              double lost_error) {
  
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
    
    DEBUG_ASSERT(mu_nu->stat().remaining_references() ==
                 mu_nu->left()->stat().remaining_references());
    
    FillApproximationCoulomb_(mu_nu->left(), rho_sigma, 
                              integral_approximation, lost_error);
    FillApproximationCoulomb_(mu_nu->right(), rho_sigma, 
                              integral_approximation, lost_error);
    
    PropagateBoundsUp_(mu_nu);
    
  }
  // is this necessary?
  /*
  else if (RectangleOnDiagonal_(rho, sigma)) {
    
    FillApproximationCoulomb_(mu_nu, rho_sigma->left(), 
                              integral_approximation, lost_error);
    FillApproximationCoulomb_(mu_nu, rho_sigma->right(), 
                              integral_approximation, lost_error);
    
    // Because the approximation has been counted twice
    // everything else has been done twice too
    mu_nu->stat().set_entry_upper_bound(mu_nu->stat().entry_upper_bound() - 
                                        integral_approximation);
    mu_nu->stat().set_entry_lower_bound(mu_nu->stat().entry_lower_bound() - 
                                        integral_approximation);
    
  }
   */
  else {
    
    // this may be inefficient
    // might be able to push the results down and do this in one 
    // post-processing step 
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
    
    mu_nu->stat().set_relative_error_bound(mu_nu->stat().entry_lower_bound(),
                                           mu_nu->stat().entry_upper_bound());
    
    index_t new_refs;

    if (RectangleOnDiagonal_(rho, sigma)) {
      index_t on_diagonal = CountOnDiagonal_(rho_sigma);
      index_t off_diagonal = (rho->count() * sigma->count()) - on_diagonal;
      new_refs = on_diagonal + 2 * off_diagonal;
    }
    else if (rho == sigma){
      new_refs = rho->count() * sigma->count();
    }
    else {
      new_refs = 2 * rho->count() * sigma->count();
    }
    
    // replace this with CountOnDiagonal to get rid of extra call above
    /*
    index_t new_refs = rho->count() * sigma->count();
    
    if (rho != sigma) {
      DEBUG_ASSERT(!((rho->begin() <= sigma->begin()) && 
                     (rho->end() >= sigma->end())));
      DEBUG_ASSERT(!((rho->begin() >= sigma->begin()) && 
                     (rho->end() <= sigma->end())));
      new_refs = 2 * new_refs;
    }
    */
    
    // reduces remaining references, but never alters epsilon
    mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                           - new_refs);
    
    DEBUG_ASSERT(mu_nu->stat().remaining_references() >= 0);
    
    // this could be a problem if the relative error bound is very close to 0
    if (relative_error_ && (mu_nu->stat().relative_error_bound() != 0.0)) {
      lost_error /= mu_nu->stat().relative_error_bound();
    }
    else if (relative_error_) {
      // I think this should be true in this case anyway
      lost_error = 0.0;
    }
    
    mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon() - lost_error);

    DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
    
    mu_nu->stat().set_approximation_val(integral_approximation);
    
  }
  
} // FillApproximationCoulomb_()


void MultiTreeFock::FillApproximationExchange_(SquareTree* mu_nu, 
                                               SquareTree* rho_sigma,
                                               double integral_approximation,
                                               double lost_error) {
  
  FockTree* mu = mu_nu->query1();
  FockTree* nu = mu_nu->query2();
  FockTree* rho = rho_sigma->query1();
  FockTree* sigma = rho_sigma->query2();
  
  double this_lost_error = lost_error;
  
  if (RectangleOnDiagonal_(mu, nu)) {
    
    PropagateBoundsDown_(mu_nu);
    
    DEBUG_ASSERT(mu_nu->stat().remaining_references() ==
                 mu_nu->left()->stat().remaining_references());
    
    FillApproximationExchange_(mu_nu->left(), rho_sigma, 
                               integral_approximation, lost_error);
    FillApproximationExchange_(mu_nu->right(), rho_sigma, 
                               integral_approximation, lost_error);
    
    PropagateBoundsUp_(mu_nu);
    
  }
  // this case should never happen, since there are no prunes when rho_sigma
  // is on the diagonal
  /*
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
  */
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
    
    mu_nu->stat().set_relative_error_bound(mu_nu->stat().entry_lower_bound(),
                                           mu_nu->stat().entry_upper_bound());
    
    index_t new_refs = rho->count() * sigma->count();
    if (rho != sigma) {
      new_refs = 2 * new_refs;
    }
    
    if (relative_error_ && (mu_nu->stat().relative_error_bound() != 0.0)) {
      this_lost_error = this_lost_error / mu_nu->stat().relative_error_bound();
    }
    else if (relative_error_) {
      this_lost_error = 0.0;
    }
    
    mu_nu->stat().set_remaining_epsilon(mu_nu->stat().remaining_epsilon() 
                                        - this_lost_error);
    DEBUG_ASSERT(mu_nu->stat().remaining_epsilon() >= 0.0);
    
    mu_nu->stat().set_remaining_references(mu_nu->stat().remaining_references() 
                                           - new_refs);
    
    mu_nu->stat().set_approximation_val(integral_approximation);
    
  }
  
} // FillApproximationExchange_()


void MultiTreeFock::PropagateBoundsDown_(SquareTree* query) {
  
  if (query->left()) {
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
      
      query->left()->stat().set_relative_error_bound(query->left()->stat().entry_lower_bound(),
                                                     query->left()->stat().entry_upper_bound());
      query->right()->stat().set_relative_error_bound(query->right()->stat().entry_lower_bound(),
                                                      query->right()->stat().entry_upper_bound());
      
      query->stat().set_approximation_val(0.0);
      
    }
  }
  
} // PropagateBoundsDown_()

void MultiTreeFock::PropagateBoundsUp_(SquareTree* query) {
  
  if (query->left() != NULL) {
  
    double min_entry = query->left()->stat().entry_lower_bound();
    double max_entry = query->left()->stat().entry_upper_bound();
    
    min_entry = min(min_entry, 
                    query->right()->stat().entry_lower_bound());
    max_entry = max(max_entry, 
                    query->right()->stat().entry_upper_bound());
    
    query->stat().set_entry_upper_bound(max_entry);
    query->stat().set_entry_lower_bound(min_entry);
    
    DEBUG_ASSERT(max_entry >= min_entry);
    
    query->stat().set_relative_error_bound(min_entry, max_entry);
    
    // Do I actually need to set this here?
    query->stat().set_remaining_references(query->left()->stat().remaining_references());
    
    DEBUG_ASSERT(query->stat().remaining_references() == 
                 query->right()->stat().remaining_references());
    
    // should I mess with the remaining epsilon in this function?
    
    double min_error = query->left()->stat().remaining_epsilon();
    min_error = min(min_error, query->right()->stat().remaining_epsilon());
    query->stat().set_remaining_epsilon(min_error);
    
  }
  
} // PropagateBoundsUp_()


void MultiTreeFock::ComputeCoulombRecursion_(SquareTree* query, 
                                             SquareTree* reference) {
  
  DEBUG_ASSERT(query->query1()->end() > query->query2()->begin());
  DEBUG_ASSERT(reference->query1()->end() > reference->query2()->begin());
  
  double integral_approximation;
  double lost_error;
  
  // should I check pruning for the leaf as well?
  if (query->is_leaf() && reference->is_leaf()) {
    
    coulomb_base_cases_++;
    ComputeCoulombBaseCase_(query, reference);
    
  }
  else if(CanApproximateCoulomb_(query, reference, &integral_approximation,
                                 &lost_error)) {
    
    DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
    DEBUG_ASSERT(lost_error != BIG_BAD_NUMBER);
    
    coulomb_approximations_++;
    
    FillApproximationCoulomb_(query, reference, integral_approximation, 
                              lost_error);
    
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
  double lost_error;
  
  if (query->is_leaf() && reference->is_leaf()) {
    
    exchange_base_cases_++;
    ComputeExchangeBaseCase_(query, reference);
    
  }
  else if (CanApproximateExchange_(query, reference, &integral_approximation,
                                   &lost_error)) {
    
    DEBUG_ASSERT(integral_approximation != BIG_BAD_NUMBER);
    
    exchange_approximations_++;
    
    FillApproximationExchange_(query, reference, integral_approximation, 
                               lost_error);
    
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
void MultiTreeFock::SetEntryBounds_(SquareTree* root) {
  
  double entry_upper;
  double entry_lower;
  
  if (root->left() == NULL) {
    double density_upper = root->stat().density_upper_bound();
    double density_lower = root->stat().density_lower_bound();
    
    double exp_upper = max(root->query1()->stat().max_bandwidth(),
                           root->query2()->stat().max_bandwidth());
    double exp_lower = min(root->query1()->stat().min_bandwidth(),
                           root->query2()->stat().min_bandwidth());
    
    DEBUG_ASSERT(density_upper >= density_lower);
    DEBUG_ASSERT(exp_upper >= exp_lower);
    
    
    double max_dist = 
        root->query1()->bound().MaxDistanceSq(root->query2()->bound());
    double min_dist = root->query1()->bound().MinDistanceSq(root->query2()->bound());
        
    double max_norm = eri::ComputeNormalization(exp_upper, 0);
    double min_norm = eri::ComputeNormalization(exp_lower, 0);
        
    // these need to be global min and max bandwidths
    double max_integral = eri::DistanceIntegral(exp_lower, exp_lower, exp_lower, 
                                                exp_lower, min_dist, 0.0, 0.0);
    max_integral *= pow(max_norm, 4.0);
                                                
    // these need to be global maxima
    double min_integral = eri::DistanceIntegral(exp_upper, exp_upper, exp_upper, 
                                                exp_upper, max_dist, 
                                                max_dist, max_dist);
    min_integral *= pow(min_norm, 4.0);
    
    DEBUG_ASSERT(max_integral >= min_integral);
    
    // does this overcount the diagonal?
    entry_upper = max_integral * number_of_basis_functions_ * number_of_basis_functions_;
    entry_lower = min_integral * number_of_basis_functions_ * number_of_basis_functions_;
    
    DensityFactor_(&entry_upper, &entry_lower, density_upper, density_lower);
    
    DEBUG_ASSERT(entry_upper >= entry_lower);
    
  }
  else {
    entry_lower = min(root->left()->stat().entry_lower_bound(),
                      root->right()->stat().entry_lower_bound());
    entry_upper = max(root->left()->stat().entry_upper_bound(),
                      root->right()->stat().entry_upper_bound());
    
    DEBUG_ASSERT(entry_upper >= entry_lower);
  }
  
  
  root->stat().set_entry_upper_bound(entry_upper);
  root->stat().set_entry_lower_bound(entry_lower);
  
  root->stat().set_relative_error_bound(entry_lower, entry_upper);
  
  if ((fabs(entry_upper - entry_lower) < 10e-10) && entry_upper != 0.0) {
    printf("Equal Bounds.\n");
  }
  
} // SetEntryBounds_


void MultiTreeFock::ResetTreeForExchange_(SquareTree* root) {
  
  if (root->left() != NULL) {
    
    ResetTreeForExchange_(root->left());
    ResetTreeForExchange_(root->right());
    
  }
  
  SetEntryBounds_(root);

  root->stat().set_remaining_references(number_of_basis_functions_ * 
                                        number_of_basis_functions_);
  
  root->stat().set_remaining_epsilon((1.0 - epsilon_split_) * epsilon_);
  
  root->stat().set_approximation_val(0.0);
  
  
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
  
  SetEntryBounds_(root);
  
  root->stat().set_remaining_epsilon(epsilon_ * epsilon_split_);
  
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



///////////////////// public functions ////////////////////////////////////

void MultiTreeFock::Compute() {

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
  //SetEntryBounds_(square_tree_);
  // set entry bounds is now covered in resetting the tree
  ResetTreeForExchange_(square_tree_);
  fx_timer_start(module_, "exchange_recursion");
  ComputeExchangeRecursion_(square_tree_, square_tree_);
  fx_timer_stop(module_, "exchange_recursion");
    
  la::SubOverwrite(exchange_matrix_, coulomb_matrix_, &fock_matrix_);
  
  fx_timer_stop(module_, "multi_time");
  
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
  fx_result_int(module_, "num_schwartz_prunes", num_schwartz_prunes_);
  fx_result_int(module_, "num_integrals_computed", num_integrals_computed_);

} // ComputeFockMatrix()

void MultiTreeFock::UpdateDensity(const Matrix& new_density) {
  
  density_matrix_.CopyValues(new_density);
  // this won't be correct when I switch to higher momentum
  ApplyPermutation(old_from_new_centers_, &density_matrix_);
  
  //density_matrix_.PrintDebug();
  
  // Reset tree density bounds
  ResetTree_(square_tree_);
  
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

void MultiTreeFock::OutputCoulomb(Matrix* coulomb_out) {
  
  if (coulomb_out) {
    coulomb_out->Copy(coulomb_matrix_);
    UnApplyPermutation(old_from_new_centers_, coulomb_out);
  }
  
} // OutputCoulomb

void MultiTreeFock::OutputExchange(Matrix* exchange_out) {
  
  if (exchange_out) {
    exchange_out->Copy(exchange_matrix_);
    UnApplyPermutation(old_from_new_centers_, exchange_out);
  }
  
} // OutputExchange




