/**
 * @file kernel_derivative.h
 *
 * The header file for the class for computing derivatives of kernel functions
 */

#ifndef KERNEL_DERIVATIVE
#define KERNEL_DERIVATIVE

#include "fastlib/fastlib.h"

/**
 * Derivative computer class for Gaussian kernel
 */
class GaussianKernelDerivative {
  FORBID_COPY(GaussianKernelDerivative);

 public:

  GaussianKernelDerivative() {}

  ~GaussianKernelDerivative() {}

  double BandwidthFactor(double bandwidth_sq) const {
    return sqrt(2 * bandwidth_sq);
  }

  void ComputeDirectionalDerivatives(const Vector &x, 
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

  double ComputePartialDerivative(const Matrix &derivative_map,
				  ArrayList<int> mapping) const {
    
    double partial_derivative = 1.0;
    
    for(index_t d = 0; d < mapping.size(); d++) {
      partial_derivative *= derivative_map.get(d, mapping[d]);
    }
    return partial_derivative;
  }

};

/**
 * Derivative computer class for Epanechnikov kernel
 */
class EpanKernelDerivative {
  FORBID_COPY(EpanKernelDerivative);
  
 public:

  EpanKernelDerivative() {}

  ~EpanKernelDerivative() {}

  double BandwidthFactor(double bandwidth_sq) const {
    return sqrt(bandwidth_sq);
  }

  void ComputeDirectionalDerivatives(const Vector &x, 
				     Matrix &derivative_map) const {

    int dim = derivative_map.n_rows();
    int order = derivative_map.n_cols() - 1;
    
    // precompute necessary Hermite polynomials based on coordinate difference
    for(index_t d = 0; d < dim; d++) {
      
      double coord_div_band = x[d];
      
      derivative_map.set(d, 0, 1 - coord_div_band * coord_div_band);

      if(order > 0) {
	derivative_map.set(d, 1, 2 * coord_div_band);
	
	if(order > 1) {
	  derivative_map.set(d, 2, -2);

	  for(index_t k = 3; k <= order; k++) {
	    derivative_map.set(d, k, 0);
	  }
	}
      }

    } // end of looping over each dimension
  }

  double ComputePartialDerivative(const Matrix &derivative_map,
				  ArrayList<int> mapping) const {
    
    int nonzero_count = 0;
    int nonzero_index = 0;

    for(index_t d = 0; d < mapping.size(); d++) {
      if(mapping[d] > 0) {
	nonzero_count++;
	nonzero_index = d;
      }

      if(nonzero_count > 1) {
	return 0;
      }
    }
    return derivative_map.get(nonzero_index, mapping[nonzero_index]);
  }

};

#endif
