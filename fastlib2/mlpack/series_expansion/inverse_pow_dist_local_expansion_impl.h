#ifndef INSIDE_INVERSE_POW_DIST_LOCAL_EXPANSION_H
#error "This is not a public header file!"
#endif

#ifndef INVERSE_POW_DIST_LOCAL_EXPANSION_IMPL_H
#define INVERSE_POW_DIST_LOCAL_EXPANSION_IMPL_H

void InversePowDistLocalExpansion::Accumulate(const Vector &v, 
					      double weight, int order) {

  // Convert the coordinates of v into ones with respect to the center
  // of expansion.
  double x_coord = v[0] - center_[0];
  double y_coord = v[1] - center_[1];
  double z_coord = v[2] - center_[2];
  double magnitude_of_vector_in_xy_plane =
    sqrt(math::Sqr(x_coord) + math::Sqr(y_coord));
  std::complex<double> eta(x_coord / magnitude_of_vector_in_xy_plane,
			   -y_coord / magnitude_of_vector_in_xy_plane);
  std::complex<double> xi(x_coord / magnitude_of_vector_in_xy_plane,
			  y_coord / magnitude_of_vector_in_xy_plane);

  // Temporary variables used for exponentiation.
  std::complex<double> power_of_eta(0.0, 0.0);
  std::complex<double> power_of_xi(0.0, 0.0);
  std::complex<double> contribution(0.0, 0.0);

  for(index_t n = 0; n <= order; n++) {
    
    // Reference to the matrix to be updated.
    GenMatrix<std::complex<double> > &n_th_order_matrix = coeffs_[n];

    for(index_t a = 0; a <= n; a++) {
      
      // $(z_i)^{n - a}$
      double power_of_z_coord = pow(z_coord, n - a);

      for(index_t b = 0; b <= a; b++) {

	InversePowDistSeriesExpansionAux::PowWithRootOfUnity
	  (eta, b, power_of_eta);
	power_of_eta *= pow(magnitude_of_vector_in_xy_plane, b);

	InversePowDistSeriesExpansionAux::PowWithRootOfUnity
	  (xi, a - b, power_of_xi);
	power_of_xi *= pow(magnitude_of_vector_in_xy_plane, a - b);

	contribution = weight * power_of_z_coord * power_of_eta * power_of_xi;
	n_th_order_matrix.set(a, b, n_th_order_matrix.get(a, b) +
			      contribution);
      }
    }
  } // end of iterating over each order...

  // Set the order to the max of the given order and the current
  // order.
  order_ = std::max(order_, order);
}

void InversePowDistLocalExpansion::AccumulateCoeffs(const Matrix& data, 
						    const Vector& weights, 
						    int begin, int end, 
						    int order) {
  for(index_t p = begin; p < end; p++) {
    Vector column_vector;
    data.MakeColumnVector(p, &column_vector);
    Accumulate(column_vector, weights[begin], order);
  }
}

template<typename TKernelAux>
void InversePowDistLocalExpansion::PrintDebug(const char *name, 
					      FILE *stream) const {

}

#endif
