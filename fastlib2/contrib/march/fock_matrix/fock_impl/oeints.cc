#include "oeints.h"


namespace oeints {

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