#include "oeints.h"


namespace oeints {

  double ComputeOverlapIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B) {
    
    /*
    Vector p_vec;
    double gamma =  eri::ComputeGPTCenter(center_A, exp_A, center_B, exp_B, 
                                          &p_vec);
    */
    double gamma = exp_A + exp_B;
     
    double dist_sq = la::DistanceSqEuclidean(center_A, center_B);
    
    double gpt = eri::IntegralGPTFactor(exp_A, exp_B, dist_sq);
    
    double prefactor = pow((math::PI/gamma), 1.5);
    
    // momentum stuff goes here later
    
    double normalization_A = eri::ComputeNormalization(exp_A, mom_A);
    double normalization_B = eri::ComputeNormalization(exp_B, mom_B);
    
    return (normalization_A * normalization_B * prefactor * gpt);
    
  }
  
  /**
   * Only works for S-overlap
   *
   * This needs to return up to 9 integrals
   */
  double ComputeOverlapIntegral(BasisShell& shellA, BasisShell& shellB) {
  
    Vector p_vec;
    double gamma = eri::ComputeGPTCenter(shellA.center(), shellA.exp(), 
                                         shellB.center(), shellB.exp(), &p_vec);
  
    double prefactor = pow((math::PI/gamma), 1.5);
    
    double dist_sq = la::DistanceSqEuclidean(shellA.center(), shellB.center());
    
    double gpt = eri::IntegralGPTFactor(shellA.exp(), shellB.exp(), dist_sq);
    
    // compute the momentum parts
    // only the three integrals where the cartesian coordinates match up will
    // have an extra factor here
    
    /*
    double momentum_fac_x = (p_vec[0] - shellA.center()[0]);
    momentum_fac_x += (p_vec[0] - shellB.center()[0]);
    momentum_fac_x /= (2 * gamma);
    
    double momentum_fac_y = (p_vec[1] - shellA.center()[1]);
    momentum_fac_y += (p_vec[1] - shellB.center()[1]);
    momentum_fac_y /= (2 * gamma);

    double momentum_fac_z = (p_vec[2] - shellA.center()[2]);
    momentum_fac_z += (p_vec[2] - shellB.center()[2]);
    momentum_fac_z /= (2 * gamma);
    */
    
    // make an array of integrals or something here
    
    // should probably normalize in here
    return (gpt * prefactor * shellA.normalization_constant() 
            * shellB.normalization_constant());
  
  } // ComputeOverlapIntegral
  
  // only works with s-type functions for now
  double ComputeKineticIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B) {
    
    double normalization_A = eri::ComputeNormalization(exp_A, mom_A);
    double normalization_B = eri::ComputeNormalization(exp_B, mom_B);
    
    double dist_sq = la::DistanceSqEuclidean(center_A, center_B);
    
    double gamma = exp_A + exp_B;
    
    double gpt = eri::IntegralGPTFactor(exp_A, exp_B, dist_sq);
    
    double prefac = 3 * exp_A * exp_B / gamma;
    prefac = prefac - (2 * dist_sq * exp_A*exp_A 
                       * exp_B*exp_B/(gamma * gamma));
    
    double integral = pow((math::PI/gamma), 1.5);
    integral *= normalization_A * normalization_B;
    integral *= gpt * prefac;
    
    return integral;
    
  } // ComputeKineticIntegral

  
  double ComputeKineticIntegral(BasisShell& shellA, BasisShell& shellB) {
    
    double integral;
    
    double AB_dist_sq = la::DistanceSqEuclidean(shellA.center(), shellB.center());
    double gamma = shellA.exp() + shellB.exp();
    
    double gpt = eri::IntegralGPTFactor(shellA.exp(), shellB.exp(), AB_dist_sq);
    
    integral = pow((math::PI/gamma), 1.5);
    integral *= gpt;
    
    double prefac = 3 * shellA.exp() * shellB.exp() / gamma;
    prefac = prefac - (2 * AB_dist_sq * shellA.exp()*shellA.exp() 
                       * shellB.exp()*shellB.exp()/(gamma * gamma));
    
    integral *= prefac;
    
    integral *= shellA.normalization_constant() * shellB.normalization_constant();
    
    return integral;
    
  }
  
  double ComputeNuclearIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B, 
                                const Vector& nuclear_center, 
                                int nuclear_charge) {
    
    Vector p_vec;
    double gamma = eri::ComputeGPTCenter(center_A, exp_A, 
                                         center_B, exp_B, &p_vec);
    
    double integral = 2 * math::PI / gamma;
    
    double AB_dist_sq = la::DistanceSqEuclidean(center_A, center_B);
    double gpt = eri::IntegralGPTFactor(exp_A, exp_B, AB_dist_sq);
    
    integral *= gpt;
    
    double CP_dist_sq = la::DistanceSqEuclidean(p_vec, nuclear_center);
    double f_part = eri::F_0_(CP_dist_sq * gamma);
    integral *= f_part;
    
    integral *= eri::ComputeNormalization(exp_A, mom_A);
    integral *= eri::ComputeNormalization(exp_B, mom_B);
    integral *= nuclear_charge;
    
    return integral;
    
  } // ComputeNuclearIntegral
  
  double ComputeNuclearIntegral(BasisShell& shellA, BasisShell& shellB, 
                                const Vector& nuclear_center) {
   
    double integral;
    
    Vector p_vec;
    double gamma = eri::ComputeGPTCenter(shellA.center(), shellA.exp(), 
                                         shellB.center(), shellB.exp(), &p_vec);
    
    double prefac = 2 * math::PI / gamma;
    integral = prefac;
    
    double AB_dist_sq = la::DistanceSqEuclidean(shellA.center(), shellB.center());
    double gpt = eri::IntegralGPTFactor(shellA.exp(), shellB.exp(), AB_dist_sq);
    
    integral *= gpt;
    
    double CP_dist_sq = la::DistanceSqEuclidean(p_vec, nuclear_center);
    double f_part = eri::F_0_(CP_dist_sq * gamma);
    integral *= f_part;
    
    integral *= shellA.normalization_constant();
    integral *= shellB.normalization_constant();
    
    return integral;
    
  }
  

};