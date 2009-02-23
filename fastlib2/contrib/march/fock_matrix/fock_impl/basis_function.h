#ifndef BASIS_FUNCTION_H
#define BASIS_FUNCTION_H

#include "fastlib/fastlib.h"


/**
 * A single (contracted) Gaussian basis function
 */
class BasisFunction {

 public:
 
  BasisFunction() {}
  
  ~BasisFunction() {}
  
  void Init() {
  
    
  
  } // Init()
 
  index_t zeta() {
    return zeta_;
  }
  
  double exponent(index_t i) {
  
    DEBUG_ASSERT(i < zeta_);
    
    return exponents_[i];
  
  }

  double coefficient(index_t i) {
    
    DEBUG_ASSERT(i < zeta_);
    
    return coefficients_[i];
    
  }
  
  const Vector& center() {
    return center_;
  }
  
 
 private:
  
  // The degree of contraction
  index_t zeta_;
  
  ArrayList<double> exponents_;
  
  ArrayList<double> coefficients_;
  
  Vector center_;
  
  // The normalization constant
  double normalization_;
  
  
  
}; // class BasisFunction

#endif
