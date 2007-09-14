/**
 * @file kernel_derivative.h
 *
 * The header file for the class for computing derivatives of kernel functions
 */

#ifndef KERNEL_DERIVATIVE
#define KERNEL_DERIVATIVE

#include "fastlib/fastlib.h"

/**
 * Derivative computer class
 */
class GaussianKernelDerivative {
  FORBID_COPY(GaussianKernelDerivative);
  
 public:

  GaussianKernelDerivative() {}

  ~GaussianKernelDerivative() {}

  void ComputePartialDerivatives(const Vector &x, 
				 Matrix &derivative_map) const {
    
    int dim = derivative_map.n_rows();
    int order = derivative_map.n_cols() - 1;
    
    // precompute necessary Hermite polynomials based on coordinate difference
    for(index_t d = 0; d < dim; d++) {
      
      double coord_div_band = x[d];
      double d2 = 2 * coord_div_band;
      double facj = exp(-coord_div_band * coord_div_band);
      
      derivative_map.set(d, 0, facj);
      
      if(order > 0) {
	
	derivative_map.set(d, 1, d2 * facj);
	
	if(order > 1) {
	  for(index_t k = 1; k < order; k++) {
	    int k2 = k * 2;
	    derivative_map.set(d, k + 1, d2 * derivative_map.get(d, k) -
			       k2 * derivative_map.get(d, k - 1));
	  }
	}
      }
    } // end of looping over each dimension
  }
};

#endif
