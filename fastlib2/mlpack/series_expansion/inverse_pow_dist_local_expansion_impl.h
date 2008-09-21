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
  double magnitude_of_vector_in_xy_plane =
    sqrt(math::Sqr(x_diff) + math::Sqr(y_diff));
  std::complex<double> eta(x_diff / magnitude_of_vector_in_xy_plane,
			   -y_diff / magnitude_of_vector_in_xy_plane);
  std::complex<double> xi(x_diff / magnitude_of_vector_in_xy_plane,
			  y_diff / magnitude_of_vector_in_xy_plane);

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
	power_of_eta *= pow(magnitude_of_vector_in_xy_plane, b);

	InversePowDistSeriesExpansionAux::PowWithRootOfUnity
	  (xi, a - b, power_of_xi);
	power_of_xi *= pow(magnitude_of_vector_in_xy_plane, a - b);
	
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
  
  /*
  // Get the pointer to the destination local moments.
  const ArrayList<GenMatrix<std::complex<double> > > *dest_local_coeffs =
    se.get_coeffs();

  // Get the pointer to the multiplicative constants.
  const ArrayList<Matrix> *multiplicative_constants =
    sea_->get_multiplicative_constants();

  // Compute the centered difference between the new center and the
  // old center.
  const Vector *new_center = se.get_center();
  double x_diff = (new_center_[0] - center_[0]);
  double y_diff = (new_center_[1] - center_[1]);
  double z_diff = (new_center_[2] - center_[2]);
  double magnitude_of_vector_in_xy_plane =
    sqrt(math::Sqr(x_diff) + math::Sqr(y_diff));
  std::complex<double> eta(x_diff / magnitude_of_vector_in_xy_plane,
                           -y_diff / magnitude_of_vector_in_xy_plane);
  std::complex<double> xi(x_diff / magnitude_of_vector_in_xy_plane,
                          y_diff / magnitude_of_vector_in_xy_plane);

  // Temporary variables used for exponentiation.
  std::complex<double> power_of_eta(0.0, 0.0);
  std::complex<double> power_of_xi(0.0, 0.0);
  std::complex<double> contribution(0.0, 0.0);

  for(index_t n_prime = 0; n_prime <= se.get_order(); n_prime++) {

    // Get the matrix reference to the coefficients to be stored.
    GenMatrix<std::complex<double> > &nprime_th_order_destination_matrix =
      coeffs_[n_prime];

    // Get the $n'$-th matrix reference to the multiplicative constants.
    const Matrix &n_prime_th_order_multiplicative_constants =
      (*multiplicative_constants)[n_prime];

    for(index_t a_prime = 0; a_prime <= n_prime; a_prime++) {
      for(index_t b_prime = 0; b_prime <= a_prime; b_prime++) {
        for(index_t n = 0; n <= n_prime; n++) {

          index_t l_a = std::max(0, a_prime + n - n_prime);
          index_t u_a = std::min(n, a_prime);

          // Get the matrix reference to the coefficients to be
          // translated.
          const GenMatrix<std::complex<double> > &n_th_order_source_matrix =
            (*coeffs_to_be_translated)[n];

          // Get the matrix reference to the multiplicative constants.
          const Matrix &n_th_order_multiplicative_constants =
            (*multiplicative_constants)[n];

          // Get the matrix reference to the multiplicative constants.
          const Matrix &nprime_minus_n_th_order_multiplicative_constants =
            (*multiplicative_constants)[n_prime - n];

          for(index_t a = l_a; a <= u_a; a++) {

            // $(z_i)^{n' - n - a' + a}$
            double power_of_z_coord = pow(z_diff, n_prime - n - a_prime + a);

            index_t l_b = std::max(0, b_prime + a - a_prime);
            index_t u_b = std::min(a, b_prime);

            for(index_t b = l_b; b <= u_b; b++) {

	      InversePowDistSeriesExpansionAux::PowWithRootOfUnity
                (eta, b_prime - b, power_of_eta);
              power_of_eta *= pow(magnitude_of_vector_in_xy_plane,
                                  b_prime - b);

	      InversePowDistSeriesExpansionAux::PowWithRootOfUnity
                (xi, a_prime - a - b_prime + b, power_of_xi);
              power_of_xi *= pow(magnitude_of_vector_in_xy_plane,
                                 a_prime - a - b_prime + b);

              // Add the contribution.
	      std::complex<double> contribution =
                n_th_order_source_matrix.get(a, b) *
                nprime_minus_n_th_order_multiplicative_constants.get
                (a_prime - a, b_prime - b) /
                n_prime_th_order_multiplicative_constants.get
                (a_prime, b_prime) *
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
  */

  // Set the order to the max of the current order and given order.
  order_ = std::max(order_, se.get_order());
}

#endif
