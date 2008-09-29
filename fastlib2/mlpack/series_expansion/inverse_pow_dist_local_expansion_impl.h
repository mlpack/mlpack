#ifndef INSIDE_INVERSE_POW_DIST_LOCAL_EXPANSION_H
#error "This is not a public header file!"
#endif

#ifndef INVERSE_POW_DIST_LOCAL_EXPANSION_IMPL_H
#define INVERSE_POW_DIST_LOCAL_EXPANSION_IMPL_H

void InversePowDistLocalExpansion::Accumulate(const double *v, 
					      double weight, int order) {

  // Convert the coordinates of v into ones with respect to the center
  // of expansion and ultimately into its spherical coordinate form.
  double x_diff = v[0] - center_[0];
  double y_diff = v[1] - center_[1];
  double z_diff = v[2] - center_[2];
  double radius, theta, phi;
  Matrix evaluated_polynomials;
  std::complex<double> partial_derivative;
  evaluated_polynomials.Init(order + 1, order + 1);
  InversePowDistSeriesExpansionAux::ConvertCartesianToSpherical
    (x_diff, y_diff, z_diff, &radius, &theta, &phi);
  sea_->GegenbauerPolynomials(cos(theta), evaluated_polynomials);

  // Retrieve the multiplicative factors.
  const ArrayList<Matrix> *multiplicative_constants =
    sea_->get_multiplicative_constants();

  // Iterate and add the contribution given by the current point.
  for(index_t n = 0; n <= order; n++) {

    // Reference to the n-th matrix.
    GenMatrix< std::complex<double> > &n_th_order_matrix = coeffs_[n];
    const Matrix &n_th_multiplicative_constants = 
      (*multiplicative_constants)[n];

    for(index_t a = 0; a <= n; a++) {
      for(index_t b = 0; b <= a; b++) {
	sea_->ComputePartialDerivativeFactor(n, a, b, radius, theta, phi, 
					     evaluated_polynomials,
					     partial_derivative);
	std::complex<double> product = weight *
	  n_th_multiplicative_constants.get(a, b) * partial_derivative;
	n_th_order_matrix.set(a, b, n_th_order_matrix.get(a, b) +
			      product);
      }
    }
  }

  // Set the order to the max of the given order and the current
  // order.
  order_ = std::max(order_, order);
}

void InversePowDistLocalExpansion::AccumulateCoeffs(const Matrix& data, 
						    const Vector& weights, 
						    int begin, int end, 
						    int order) {
  for(index_t p = begin; p < end; p++) {
    Accumulate(data.GetColumnPtr(p), weights[begin], order);
  }
}

double InversePowDistLocalExpansion::EvaluateField(const double *v,
						   int order) const {
  
  // The final sum.
  double result = 0;

  double x_diff = v[0] - center_[0];
  double y_diff = v[1] - center_[1];
  double z_diff = v[2] - center_[2];
  double magnitude_of_vector_in_xy_plane;
  std::complex<double> eta;
  std::complex<double> xi;
  InversePowDistSeriesExpansionAux::ConvertToComplexForm
    (x_diff, y_diff, magnitude_of_vector_in_xy_plane, eta, xi);
  
  // Temporary variables used for exponentiation.
  std::complex<double> power_of_eta(0.0, 0.0);
  std::complex<double> power_of_xi(0.0, 0.0);
    
  for(index_t n = 0; n <= order; n++) {
    
    const GenMatrix< std::complex<double> > &n_th_order_matrix = coeffs_[n];

    for(index_t a = 0; a <= n; a++) {

      // $(z_i)^{n - a}$
      double power_of_z_coord = pow(z_diff, n - a);

      for(index_t b = 0; b <= a; b++) {
	InversePowDistSeriesExpansionAux::PowWithRootOfUnity
	  (eta, b, power_of_eta);

	if(magnitude_of_vector_in_xy_plane == 0) {
	  power_of_eta.real() = power_of_eta.imag() = 0;
	}
	else {
	  power_of_eta *= pow(magnitude_of_vector_in_xy_plane, b);
	}

	InversePowDistSeriesExpansionAux::PowWithRootOfUnity
	  (xi, a - b, power_of_xi);
	
	if(magnitude_of_vector_in_xy_plane == 0) {
	  power_of_xi.real() = power_of_xi.imag() = 0;
	}
	else {
	  power_of_xi *= pow(magnitude_of_vector_in_xy_plane, a - b);
	}
	
	std::complex<double> product = n_th_order_matrix.get(a, b) *
	  power_of_eta * power_of_xi;
	
	result += power_of_z_coord * product.real();
      }
    }
  }

  return result;
}

double InversePowDistLocalExpansion::EvaluateField
(const Matrix& data, int row_num, int order) const {
  return EvaluateField(data.GetColumnPtr(row_num), order);
}

void InversePowDistLocalExpansion::Init
(const Vector &center, InversePowDistSeriesExpansionAux *sea) {

  // Copy the center.
  center_.Copy(center);

  // Set the pointer to the auxiliary object.
  sea_ = sea;

  // Initialize the order of approximation.
  order_ = -1;

  // Allocate the space for storing the coefficients.
  coeffs_.Init(sea_->get_max_order() + 1);
  for(index_t n = 0; n <= sea_->get_max_order(); n++) {
    coeffs_[n].Init(n + 1, n + 1);
    for(index_t j = 0; j <= n; j++) {
      for(index_t k = 0; k <= n; k++) {
	coeffs_[n].set(j, k, std::complex<double>(0, 0));
      }
    }
  } 
}

void InversePowDistLocalExpansion::PrintDebug(const char *name, 
					      FILE *stream) const {

}

void InversePowDistLocalExpansion::TranslateToLocal
(InversePowDistLocalExpansion &se) {

  // Get the pointer to the destination local moments.
  ArrayList<GenMatrix<std::complex<double> > > *dest_local_coeffs =
    se.get_coeffs();

  // Get the pointer to the multiplicative constants.
  const ArrayList<Matrix> *multiplicative_constants =
    sea_->get_multiplicative_constants();

  // Compute the centered difference between the new center and the
  // old center.
  const Vector *new_center = se.get_center();
  double x_diff = ((*new_center)[0] - center_[0]);
  double y_diff = ((*new_center)[1] - center_[1]);
  double z_diff = ((*new_center)[2] - center_[2]);
  double magnitude_of_vector_in_xy_plane;
  std::complex<double> eta;
  std::complex<double> xi;
  InversePowDistSeriesExpansionAux::ConvertToComplexForm
    (x_diff, y_diff, magnitude_of_vector_in_xy_plane, eta, xi);

  // Temporary variables used for exponentiation.
  std::complex<double> power_of_eta(0.0, 0.0);
  std::complex<double> power_of_xi(0.0, 0.0);
  std::complex<double> contribution(0.0, 0.0);

  for(index_t n_prime = 0; n_prime <= order_; n_prime++) {

    // Get the matrix reference to the coefficients to be stored.
    GenMatrix<std::complex<double> > &nprime_th_order_destination_matrix =
      (*dest_local_coeffs)[n_prime];

    // Get the $n'$-th matrix reference to the multiplicative constants.
    const Matrix &n_prime_th_order_multiplicative_constants =
      (*multiplicative_constants)[n_prime];

    for(index_t a_prime = 0; a_prime <= n_prime; a_prime++) {
      for(index_t b_prime = 0; b_prime <= a_prime; b_prime++) {
        for(index_t n = n_prime; n <= order_; n++) {

          // Get the matrix reference to the coefficients to be
          // translated.
          const GenMatrix<std::complex<double> > &n_th_order_source_matrix =
            coeffs_[n];

          // Get the matrix reference to the multiplicative constants.
          const Matrix &n_minus_n_prime_th_order_multiplicative_constants =
            (*multiplicative_constants)[n - n_prime];
	  const Matrix &n_th_order_multiplicative_constants =
	    (*multiplicative_constants)[n];

          for(index_t a = a_prime; a <= n - n_prime + a_prime; a++) {

            // $(z_i)^{n - n' - a + a'}$
            double power_of_z_coord = pow(z_diff, n - n_prime - a + a_prime);

            for(index_t b = b_prime; b <= a - a_prime + b_prime; b++) {

	      InversePowDistSeriesExpansionAux::PowWithRootOfUnity
                (eta, b - b_prime, power_of_eta);
              power_of_eta *= pow(magnitude_of_vector_in_xy_plane,
                                  b - b_prime);

	      InversePowDistSeriesExpansionAux::PowWithRootOfUnity
                (xi, a - a_prime - b + b_prime, power_of_xi);
              power_of_xi *= pow(magnitude_of_vector_in_xy_plane,
                                 a - a_prime - b + b_prime);

              // Add the contribution.
	      std::complex<double> contribution =
                n_th_order_source_matrix.get(a, b) *
                n_minus_n_prime_th_order_multiplicative_constants.get
                (a - a_prime, b - b_prime) *
                n_prime_th_order_multiplicative_constants.get
                (a_prime, b_prime) /
		n_th_order_multiplicative_constants.get(a, b) *
                power_of_z_coord * power_of_eta * power_of_xi;
              nprime_th_order_destination_matrix.set
                (a_prime, b_prime, nprime_th_order_destination_matrix.get
                 (a_prime, b_prime) + contribution);
            }
          }
        }
      }
    }
  }

  // Set the order of the destination local moments to the max of the
  // current order and given order.
  se.set_order(std::max(order_, se.get_order()));
}

#endif
