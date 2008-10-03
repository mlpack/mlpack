#ifndef INVERSE_POW_DIST_KERNEL_AUX_H
#define INVERSE_POW_DIST_KERNEL_AUX_H

#include "inverse_pow_dist_kernel.h"

/** @brief The auxilary class for $r / ||r||^{\lambda}$ kernels using
 *         $O(D^p)$ expansion.
 */
class InversePowDistGradientKernelAux {

 private:

  void SubFrom_(index_t dimension, int decrement,
		const ArrayList<int> &subtract_from, ArrayList<int> &result) {
    
    for(index_t d = 0; d < subtract_from.size(); d++) {
      if(d == dimension) {
	result[d] = subtract_from[d] - decrement;
      }
      else {
	result[d] = subtract_from[d];
      }
    }
  }

 public:
  
  typedef InversePowDistGradientKernel TKernel;
  
  typedef SeriesExpansionAux TSeriesExpansionAux;
  
  typedef FarFieldExpansion<InversePowDistGradientKernelAux> TFarFieldExpansion;

  typedef LocalExpansion<InversePowDistGradientKernelAux> TLocalExpansion;

  /** @brief The actual kernel object.
   */
  TKernel kernel_;
  
  /** @brief The actual series expansion auxilary object.
   */
  TSeriesExpansionAux sea_;

  OT_DEF_BASIC(InversePowDistGradientKernelAux) {
    OT_MY_OBJECT(kernel_);
    OT_MY_OBJECT(sea_);
  }

 public:

  void Init(double bandwidth, int max_order, int dim) {
    kernel_.Init(bandwidth);
    sea_.Init(max_order, dim);
  }

  void ComputeDirectionalDerivatives(const Vector &x, 
				     Matrix &derivative_map, int order) const {
    
    // Allocate the derivative map to be a long vector.
    derivative_map.Init(order, 1);
    derivative_map.SetZero();

    // Squared L2 norm of the vector.
    double squared_l2_norm = la::Dot(x, x);

    // Temporary variable to look for arithmetic operations on
    // multiindex.
    ArrayList<int> tmp_multiindex;
    tmp_multiindex.Init(sea_.get_dimension());

    for(index_t i = 0; i < derivative_map.n_rows(); i++) {
    
      // Contribution to the current multiindex position.
      double contribution = 0;

      // Retrieve the multiindex mapping.
      const ArrayList<int> &multiindex = sea_.get_multiindex(i);
      
      // Compute the contribution of $D_{x}^{n - e_d} \phi_{\nu,
      // d}(x)$ component for each $d$.
      for(index_t d = 0; d < v.length(); d++) {
	
	// Subtract 1 from the given dimension.
	SubFrom_(d, 1, multiindex, tmp_multiindex);
	index_t n_minus_e_d_position = 
	  sea_.ComputeMultiindexPosition(tmp_multiindex);
	if(n_minus_e_d_position >= 0) {
	  double factor = 2 * multiindex[d] * x[d];
	  if((kernel_.dimension_ == 0 && d == 1) ||
	     (kernel_.dimension_ > 0 && d == 0)) {
	    factor += (kernel_.lambda_ - 2);
	  }
	  contribution += factor * derivative_map.get(n_minus_e_d_position, 0);
	}
	
	// Subtract 2 from the given dimension.
	SubFrom_(d, 2, multiindex, tmp_multiindex);
	index_t n_minus_two_e_d_position =
	  sea_.ComputeMultiindexPosition(tmp_multiindex);
	if(n_minus_two_e_d_position >= 0) {
	  double factor = multiindex[d] * (multiindex[d] - 1);

	  if((kernel_.dimension_ == 0 && d == 1) ||
	     (kernel_.dimension_ > 0 && d == 0)) {
	    factor += (kernel_.lambda_ - 2) * (multiindex[d] - 1);
	  }

	  contribution += factor *
	    derivative_map.get(n_minus_two_e_d_position, 0);
	}
	
      } // end of iterating over each dimension.
      
      // Set the final contribution for this multiindex.
      derivative_map.set(i, 0, -contribution / squared_l2_norm);
      
    } // end of iterating over all required multiindex positions...
  }
  
  double ComputePartialDerivative(const Matrix &derivative_map,
				  const ArrayList<int> &mapping) const {
    
    return derivative_map.get(sea_.ComputeMultiindexPosition(mapping), 0);
  }

};

/** @brief The auxilary class for $1 / r^{\lambda}$ kernels using
 *         $O(D^p)$ expansion.
 */
class InversePowDistKernelAux {

 public:
  
  typedef InversePowDistKernel TKernel;
  
  typedef SeriesExpansionAux TSeriesExpansionAux;
  
  typedef FarFieldExpansion<InversePowDistKernelAux> TFarFieldExpansion;

  typedef LocalExpansion<InversePowDistKernelAux> TLocalExpansion;

  /** @brief The actual kernel object.
   */
  TKernel kernel_;

};


#endif
