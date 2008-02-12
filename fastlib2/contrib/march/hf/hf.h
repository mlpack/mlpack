/** 
 * @file hf.h
 *
 * @author Bill March (march@gatech.edu)
 * 
 * Contains classes for the Hartree-Fock implementation.  
 */

#ifndef HF_H
#define HF_H
#include <fastlib/fastlib.h>

/**
 * A class that stores the information for a contracted Gaussian basis function.
 *
 * TODO: Should this class also have functions for computations among contracted 
 * functions?
 */
class ContractedGaussian {
  
  FORBID_ACCIDENTAL_COPIES(ContractedGaussian);
  
 private:
  
  // The number of primitive Gaussians that make up this function
  index_t number_of_primitives_;
  ArrayList<double> bandwidths_;
  ArrayList<double> coefficients_;
  
  
 public:
  
  ContractedGaussian() {}
  
  ~ContractedGaussian() {}
  
  void Init(index_t num, const ArrayList<double>& band, 
            const ArrayList<double>& coeff) {
    
    number_of_primitives_ = num;
    bandwidths_.Copy(band);
    coefficients_.Copy(coeff);
    
  }
  
  
};




#endif