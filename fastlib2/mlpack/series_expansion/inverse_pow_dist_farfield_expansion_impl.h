#ifndef INSIDE_INVERSE_POW_DIST_FARFIELD_EXPANSION_H
#error "This file is not a public header file!"
#endif

#ifndef INVERSE_POW_DIST_FARFIELD_EXPANSION_IMPL_H
#define INVERSE_POW_DIST_FARFIELD_EXPANSION_IMPL_H

#include <complex>
#include <iostream>

void InversePowDistFarFieldExpansion::Accumulate(const Vector &v, 
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

void InversePowDistFarFieldExpansion::Init
(const Vector &center, InversePowDistSeriesExpansionAux *sea) {

  // Copy the center.
  center_.Copy(center);

  // Set the pointer to the auxiliary object.
  sea_ = sea;

  // Initialize the order of approximation.
  order_ = -1;

  // Allocate the space for storing the coefficients.
  coeffs_.Init(sea_->get_max_order() + 1);
  printf("Set to %d\n", sea_->get_max_order() + 1);
  for(index_t n = 0; n <= sea_->get_max_order(); n++) {
    coeffs_[n].Init(n + 1, n + 1);
    coeffs_[n].SetZero();
  } 
}

void InversePowDistFarFieldExpansion::PrintDebug
(const char *name, FILE *stream) const {

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Far field expansion\n");
  fprintf(stream, "Center: ");
  
  for (index_t i = 0; i < center_.length(); i++) {
    fprintf(stream, "%g ", center_[i]);
  }
  fprintf(stream, "\n");

  for(index_t n = 0; n <= order_; n++) {

    const GenMatrix< std::complex<double> > &n_th_order_matrix = coeffs_[n];
    
    for(index_t a = 0; a <= n; a++) {
      for(index_t b = 0; b <= n; b++) {
	std::cout << n_th_order_matrix.get(a, b) << " ";
      }
      std::cout << "\n";
    }
  }
}

#endif
