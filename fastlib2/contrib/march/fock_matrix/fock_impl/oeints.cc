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
    double momentum_fac_x = (p_vec[0] - shellA.center()[0]);
    momentum_fac_x += (p_vec[0] - shellB.center()[0]);
    momentum_fac_x /= (2 * gamma);
    
    double momentum_fac_y = (p_vec[1] - shellA.center()[1]);
    momentum_fac_y += (p_vec[1] - shellB.center()[1]);
    momentum_fac_y /= (2 * gamma);

    double momentum_fac_z = (p_vec[2] - shellA.center()[2]);
    momentum_fac_z += (p_vec[2] - shellB.center()[2]);
    momentum_fac_z /= (2 * gamma);
    
    // make an array of integrals or something here
    
    return (gpt * prefactor);
  
  } // ComputeOverlapIntegral
  
  
  
  double ComputeKineticIntegral(BasisShell& shellA, BasisShell& shellB) {
    
    return 0.0;
    
  }
  
  double ComputeNuclearIntegral(BasisShell& shellA, BasisShell& shellB, 
                                const Vector& nuclear_center, int nuclear_mass) {
   
    return 0.0;                            
    
  }
  

};