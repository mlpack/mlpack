#ifndef INSIDE_INVERSE_POW_DIST_FARFIELD_EXPANSION_H
#error "This file is not a public header file!"
#endif

#include <complex>
#include <iostream>

void InversePowDistFarFieldExpansion::Accumulate(const double *v, 
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
    Accumulate(data.GetColumnPtr(p), weights[begin], order);
  }
}

double InversePowDistFarFieldExpansion::EvaluateField(const double *v,
						      int order) const {

  double result = 0;
  std::complex<double> tmp(0, 0);
  std::complex<double> partial_derivative(0, 0);
  const ArrayList<Matrix> *multiplicative_constants = 
    sea_->get_multiplicative_constants();

  // First, complete a table of Gegenbauer polynomials for the
  // evluation point
  double x_diff = v[0] - center_[0];
  double y_diff = v[1] - center_[1];
  double z_diff = v[2] - center_[2];
  double radius, theta, phi;
  Matrix evaluated_polynomials;
  evaluated_polynomials.Init(order + 1, order + 1);
  InversePowDistSeriesExpansionAux::ConvertCartesianToSpherical
    (x_diff, y_diff, z_diff, &radius, &theta, &phi);
  sea_->GegenbauerPolynomials(cos(theta), evaluated_polynomials);
    
  for(index_t n = 0; n <= order; n++) {
    
    const GenMatrix< std::complex<double> > &n_th_order_matrix = coeffs_[n];
    const Matrix &n_th_multiplicative_constants = 
      (*multiplicative_constants)[n];

    for(index_t a = 0; a <= n; a++) {
      for(index_t b = 0; b <= a; b++) {
	sea_->ComputePartialDerivativeFactor(n, a, b, radius, theta, phi, 
					     evaluated_polynomials,
					     partial_derivative);

	std::complex<double> product = n_th_order_matrix.get(a, b) * 
	  partial_derivative;

	result += n_th_multiplicative_constants.get(a, b) * product.real();
      }
    }
  }

  return result;
}

double InversePowDistFarFieldExpansion::EvaluateField
(const Matrix& data, int row_num, int order) const {
  return EvaluateField(data.GetColumnPtr(row_num), order);
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
  for(index_t n = 0; n <= sea_->get_max_order(); n++) {
    coeffs_[n].Init(n + 1, n + 1);
    for(index_t j = 0; j <= n; j++) {
      for(index_t k = 0; k <= n; k++) {
	coeffs_[n].set(j, k, std::complex<double>(0, 0));
      }
    }
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

void InversePowDistFarFieldExpansion::TranslateFromFarField
(const InversePowDistFarFieldExpansion &se) {
  
  // Get the pointer to the far field moments to be translated.
  const ArrayList<GenMatrix<std::complex<double> > > *coeffs_to_be_translated =
    se.get_coeffs();

  // Get the pointer to the multiplicative constants.
  const ArrayList<Matrix> *multiplicative_constants = 
    sea_->get_multiplicative_constants();
  
  // Compute the centered difference between the new center and the
  // old center.
  const Vector *old_center = se.get_center();
  double x_diff = center_[0] - (*old_center)[0];
  double y_diff = center_[1] - (*old_center)[1];
  double z_diff = center_[2] - (*old_center)[2];
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
    
    // Get the matrix reference to the multiplicative constants.
    const Matrix &nprime_th_order_multiplicative_constants =
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
	  const Matrix &nprime_minus_n_th_order_multiplicative_constants =
	    (*multiplicative_constants)[n_prime - n];

	  for(index_t a = l_a; a <= u_a; a++) {

	    // $(z_i)^{n' - n - a' + a}$
	    double power_of_z_coord = pow(z_diff, n_prime - n + a_prime - a);

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
		nprime_th_order_multiplicative_constants.get
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
}

template<typename InversePowDistLocalExpansion>
void InversePowDistFarFieldExpansion::TranslateToLocal
(InversePowDistLocalExpansion &se, int truncation_order) {

  // Pointer to the local moment coefficients.
  ArrayList< GenMatrix<std::complex<double> > > *local_moments =
    se.get_coeffs();

  // Pointer to the multiplicative constants.
  const ArrayList<Matrix> *multiplicative_constants =
    sea_->get_multiplicative_constants();

  // First, complete a table of Gegenbauer polynomials for the
  // evluation point
  const Vector *local_expansion_center = se.get_center();
  double x_diff = -((*local_expansion_center)[0] - center_[0]);
  double y_diff = -((*local_expansion_center)[1] - center_[1]);
  double z_diff = -((*local_expansion_center)[2] - center_[2]);
  double radius, theta, phi;
  Matrix evaluated_polynomials;
  evaluated_polynomials.Init(2 * (truncation_order + 1), 2 * 
			     (truncation_order + 1));
  InversePowDistSeriesExpansionAux::ConvertCartesianToSpherical
    (x_diff, y_diff, z_diff, &radius, &theta, &phi);
  sea_->GegenbauerPolynomials(cos(theta), evaluated_polynomials);
  
  // Temporary variable storing the partial derivative.
  std::complex<double> partial_derivative;

  for(index_t n_prime = 0; n_prime <= truncation_order; n_prime++) {
    
    // Reference to the n-th order matrix of the local moments.
    GenMatrix<std::complex<double> > &local_n_th_order_matrix =
      (*local_moments)[n_prime];

    // Multiplicative constants corresponding to the local moment
    // index.
    const Matrix &local_n_th_multiplicative_constants = 
      (*multiplicative_constants)[n_prime];
    
    for(index_t a_prime = 0; a_prime <= n_prime; a_prime++) {
      for(index_t b_prime = 0; b_prime <= a_prime; b_prime++) {
	
	for(index_t n = 0; n <= truncation_order; n++) {

	  // Reference to the n-th order matrix of the far-field
	  // moments.
	  const GenMatrix<std::complex<double> > &farfield_n_th_order_matrix =
	    coeffs_[n];

	  // Multiplicative constants corresponding to the far-field
	  // moment index.
	  const Matrix &farfield_n_th_multiplicative_constants =
	    (*multiplicative_constants)[n];

	  for(index_t a = 0; a <= n; a++) {
	    for(index_t b = 0; b <= a; b++) {

	      sea_->ComputePartialDerivativeFactor
		(n + n_prime, a + a_prime, b + b_prime, radius, theta, phi,
		 evaluated_polynomials, partial_derivative);

	      std::complex<double> product = 
		local_n_th_multiplicative_constants.get(a_prime, b_prime) * 
		farfield_n_th_order_matrix.get(a, b) * 
		farfield_n_th_multiplicative_constants.get(a, b) * 
		partial_derivative;
	      
	      // Accumulate the local moment for this order.
	      local_n_th_order_matrix.set
		(a_prime, b_prime, 
		 local_n_th_order_matrix.get(a_prime, b_prime) + product);
	    }
	  }
	}
      }
    }
  }

  // Update the order to a higher one if necessary.
  se.set_order(std::max(se.get_order(), truncation_order));

} // end of the function...
