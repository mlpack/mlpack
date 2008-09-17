#ifndef INSIDE_INVERSE_POW_DIST_FARFIELD_EXPANSION_H
#error "This file is not a public header file!"
#endif

#ifndef INVERSE_POW_DIST_FARFIELD_EXPANSION_IMPL_H
#define INVERSE_POW_DIST_FARFIELD_EXPANSION_IMPL_H

#include <complex>

void InversePowDistFarFieldExpansion::Accumulate(const Vector &v, 
						 double weight, int order) {

  // Convert the coordinates of v into ones with respect to the center
  // of expansion.
  double x_coord = v[0] - center_[0];
  double y_coord = v[1] - center_[1];
  double z_coord = v[2] - center_[2];
  double magnitude_of_vector_in_xy_plane =
    sqrt(math::Sqr(x_coord) + math::Sqr(y_coord));
  std::complex<double> first_complex
    (x_coord / magnitude_of_vector_in_xy_plane,
     -y_coord / magnitude_of_vector_in_xy_plane);
  double first_complex_arg = atan2(-y_coord, x_coord);
  std::complex<double> second_complex
    (x_coord / magnitude_of_vector_in_xy_plane,
     y_coord / magnitude_of_vector_in_xy_plane);
  double second_complex_arg = atan2(y_coord, x_coord);

  for(index_t n = 0; n <= order; n++) {
    
    // Reference to the matrix to be updated.
    GenMatrix<std::complex<double> > &n_th_order_matrix = coeffs_[n];

    for(index_t a = 0; a <= n; a++) {
      
      // $(z_i)^{n - a}$
      double power_of_z_coord = std::pow(z_coord, n - a);

      for(index_t b = 0; b <= a; b++) {

	std::complex<double> power_of_eta = 
	  pow(magnitude_of_vector_in_xy_plane, b) *
	  std::complex<double>(cos(b * first_complex_arg), 
			       sin(b * first_complex_arg));
	std::complex<double> power_of_xi = 
	  pow(magnitude_of_vector_in_xy_plane, a - b) *
	  std::complex<double>(cos((a - b) * second_complex_arg),
			       sin((a - b) * second_complex_arg));

	std::complex<double> contribution = 
	  weight * power_of_z_coord * power_of_eta * power_of_xi;
	n_th_order_matrix.set(a, b, n_th_order_matrix.get(a, b) +
			      contribution);
      }
    }
  }
}

void InversePowDistFarFieldExpansion::AccumulateCoeffs
(const Matrix& data, const Vector& weights, int begin, int end, int order) {

  for(index_t p = begin; p < end; p++) {
    Vector column_vector;
    data.MakeColumnVector(p, &column_vector);
    Accumulate(column_vector, weights[begin], order);
  }
}

double InversePowDistFarFieldExpansion::EvaluateField
(const Matrix& data, int row_num, int order) const {

  // Please implement me!
  return 0;
}

#endif
