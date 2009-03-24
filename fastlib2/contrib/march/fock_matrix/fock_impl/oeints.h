#ifndef OEINTS_H
#define OEINTS_H

#include "fastlib/fastlib.h"
#include "eri.h"

//class BasisShell;

namespace oeints {

  double ComputeOverlapIntegral(BasisShell& shellA, BasisShell& shellB);
  
  double ComputeKineticIntegral(BasisShell& shellA, BasisShell& shellB);
  
  double ComputeNuclearIntegral(BasisShell& shellA, BasisShell& shellB, 
                                const Vector& nuclear_center, int nuclear_mass);

}; // oeints namespace


#endif