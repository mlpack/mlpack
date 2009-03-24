#include "oeints.h"


namespace oeints {

  /**
   * Only works for S-overlap
   */
  double ComputeOverlapIntegral(BasisShell& shellA, BasisShell& shellB) {
  
    double gamma = shellA.exp() + shellB.exp();
  
    double prefactor = pow((math::PI/gamma), 1.5);
    
    double dist_sq = la::DistanceSqEuclidean(shellA.center(), shellB.center());
    
    double gpt = eri::IntegralGPTFactor(shellA.exp(), shellB.exp(), dist_sq);
    
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