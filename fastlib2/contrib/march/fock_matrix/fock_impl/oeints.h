#ifndef OEINTS_H
#define OEINTS_H

#include "fastlib/fastlib.h"
#include "eri.h"

//class BasisShell;

namespace oeints {

  double ComputeOverlapIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B);
  
  double ComputeOverlapIntegral(BasisShell& shellA, BasisShell& shellB);
  
  double ComputeKineticIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B);
  
  double ComputeKineticIntegral(BasisShell& shellA, BasisShell& shellB);
  
  double ComputeNuclearIntegral(const Vector& center_A, double exp_A, int mom_A, 
                                const Vector& center_B, double exp_B, int mom_B, 
                                const Vector& nuclear_center, 
                                int nuclear_charge);
  
  double ComputeNuclearIntegral(BasisShell& shellA, BasisShell& shellB, 
                                const Vector& nuclear_center, int nuclear_mass);

}; // oeints namespace


#endif