#ifndef INSIDE_FARFIELD_EXPANSION_H
#error "This file is not a public header file!"
#endif

#ifndef FARFIELD_EXPANSION_IMPL_H
#define FARFIELD_EXPANSION_IMPL_H

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::Accumulate(const arma::vec& v, double weight,
					       int order) {

  int dim = v.n_elem;
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  arma::vec tmp;
  int r, i, j, k, t, tail;
  arma::Col<short int> heads;
  arma::vec x_r;
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  tmp.set_size(total_num_coeffs);
  heads.set_size(dim + 1);
  x_r.set_size(dim);
  arma::vec pos_coeffs;
  arma::vec neg_coeffs;
  pos_coeffs.zeros(total_num_coeffs);
  neg_coeffs.zeros(total_num_coeffs);

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  arma::vec& C_k;
    
  // Calculate the coordinate difference between the ref point and the 
  // centroid.
  for(i = 0; i < dim; i++) {
    x_r[i] = (v[i] - center_[i]) / bandwidth_factor;
  }
  
  // initialize heads
  heads.zeros();
  heads[dim] = SHRT_MAX;
  
  tmp[0] = 1.0;
  
  for(k = 1, t = 1, tail = 1; k <= order; k++, tail = t) {
    for(i = 0; i < dim; i++) {
      int head = heads[i];
      heads[i] = t;
      
      for(j = head; j < tail; j++, t++) {
	tmp[t] = tmp[j] * x_r[i];
      }
    }
  }
  
  // Tally up the result in A_k.
  for(i = 0; i < total_num_coeffs; i++) {
    double prod = weight * tmp[i];
    
    if(prod > 0) {
      pos_coeffs[i] += prod;
    }
    else {
      neg_coeffs[i] += prod;
    }
  }

  // get multiindex factors
  C_k = sea_->get_inv_multiindex_factorials();
  
  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::AccumulateCoeffs(const arma::mat& data, 
						     const arma::vec& weights, 
						     int begin, int end, 
						     int order) {
  
  int dim = data.n_rows;
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  arma::vec tmp;
  int r, i, j, k, t, tail;
  arma::Col<short int> heads;
  arma::vec x_r;
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  tmp.set_size(total_num_coeffs);
  heads.set_size(dim + 1);
  x_r.set_size(dim);
  arma::vec pos_coeffs;
  arma::vec neg_coeffs;
  pos_coeffs.zeros(total_num_coeffs);
  neg_coeffs.zeros(total_num_coeffs);

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }

  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data(i, r) - center_[i]) / bandwidth_factor;
    }

    // initialize heads
    heads.zeros();
    heads[dim] = SHRT_MAX;
    
    tmp[0] = 1.0;
    
    for(k = 1, t = 1, tail = 1; k <= order; k++, tail = t) {
      for(i = 0; i < dim; i++) {
	short int head = heads[i];
	heads[i] = t;
	
	for(j = head; j < tail; j++, t++) {
	  tmp[t] = tmp[j] * x_r[i];
	}
      }
    }
    
    // Tally up the result in A_k.
    for(i = 0; i < total_num_coeffs; i++) {
      double prod = weights[r] * tmp[i];
      
      if(prod > 0) {
	pos_coeffs[i] += prod;
      }
      else {
	neg_coeffs[i] += prod;
      }
    }
    
  } // End of looping through each reference point

  // get multiindex factors
  const arma::vec& C_k = sea_->get_inv_multiindex_factorials();

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::RefineCoeffs(const arma::mat& data, 
						 const arma::vec& weights, 
						 int begin, int end, 
						 int order) {
  
  if(order_ < 0) {
    
    AccumulateCoeffs(data, weights, begin, end, order);
    return;
  }

  int dim = data.n_rows;
  int old_total_num_coeffs = sea_->get_total_num_coeffs(order_);
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  double tmp;
  int r, i, j;
  arma::vec x_r(dim);
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  arma::vec pos_coeffs;
  arma::vec neg_coeffs;
  pos_coeffs.zeros(total_num_coeffs);
  neg_coeffs.zeros(total_num_coeffs);

  // if we already have the order of approximation, then return.
  if(order_ >= order) {
    return;
  }
  else {
    order_ = order;
  }

  const arma::vec& C_k = sea_->get_inv_multiindex_factorials();

  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {
    
    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data(i, r) - center_[i]) / bandwidth_factor;
    }

    // compute in bruteforce way
    for(i = old_total_num_coeffs; i < total_num_coeffs; i++) {
      const std::vector<short int> &mapping = sea_->get_multiindex(i);
      tmp = 1;
      
      for(j = 0; j < dim; j++) {
	tmp *= pow(x_r[j], mapping[j]);
      }

      double prod = weights[r] * tmp;
      
      if(prod > 0) {
        pos_coeffs[i] += prod;
      }
      else {
        neg_coeffs[i] += prod;
      }
    }
    
  } // End of looping through each reference point

  for(r = old_total_num_coeffs; r < total_num_coeffs; r++) {
    coeffs_[r] = (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernelAux>
double FarFieldExpansion<TKernelAux>::EvaluateField(const arma::mat& data, 
						    int row_num, 
						    int order) const {
  return EvaluateField(data.unsafe_col(row_num).memptr(), order);
}

template<typename TKernelAux>
double FarFieldExpansion<TKernelAux>::EvaluateField(const double *x_q, 
						    int order) const {
  
  // dimension
  int dim = sea_->get_dimension();

  // total number of coefficients
  int total_num_coeffs = sea_->get_total_num_coeffs(order);

  // square root times bandwidth
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());
  
  // the evaluated sum
  double pos_multipole_sum = 0;
  double neg_multipole_sum = 0;
  double multipole_sum = 0;
  
  // computed derivative map
  arma::mat derivative_map;
  ka_->AllocateDerivativeMap(dim, order, derivative_map);

  // temporary variable
  arma::vec arrtmp(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  arma::vec x_q_minus_x_R(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(index_t d = 0; d < dim; d++) {
    x_q_minus_x_R[d] = (x_q[d] - center_[d]) / bandwidth_factor;
  }

  // compute deriative maps based on coordinate difference.
  ka_->ComputeDirectionalDerivatives(x_q_minus_x_R, derivative_map, order);

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(index_t j = 0; j < total_num_coeffs; j++) {
    const std::vector<short int> &mapping = sea_->get_multiindex(j);
    double arrtmp = ka_->ComputePartialDerivative(derivative_map, mapping);
    double prod = coeffs_[j] * arrtmp;
    
    if(prod > 0) {
      pos_multipole_sum += prod;
    }
    else {
      neg_multipole_sum += prod;
    }
  }

  multipole_sum = pos_multipole_sum + neg_multipole_sum;
  return multipole_sum;
}

template<typename TKernelAux>
double FarFieldExpansion<TKernelAux>::MixField(const arma::mat &data, 
					       int node1_begin, int node1_end,
					       int node2_begin, int node2_end, 
					       const FarFieldExpansion &fe2,
					       const FarFieldExpansion &fe3, 
					       int order2, int order3) const {
  
  // bandwidth factor and multiindex mapping stuffs
  double result;
  double bandwidth_factor = ka_->BandwidthFactor(bandwidth_sq());
  const std::vector<short int>& multiindex_mapping = 
    sea_->get_multiindex_mapping();
  const std::vector<short int>& lower_mapping_index = 
    sea_->get_lower_mapping_index();

  // get the total number of coefficients and coefficients
  int total_num_coeffs2 = sea_->get_total_num_coeffs(order2);
  int total_num_coeffs3 = sea_->get_total_num_coeffs(order3);
  int dim = sea_->get_dimension();
  arma::vec& coeffs2 = fe2.get_coeffs();
  arma::vec& coeffs3 = fe3.get_coeffs();

  // actual accumulated sum
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;
  
  // some temporary
  double moment_k;
  double xi_xI, xj_xJ, diff;

  // temporary array
  std::vector<short int> beta_gamma_nu_eta_mapping;
  std::vector<short int> beta_nu_mapping;
  std::vector<short int> gamma_eta_mapping;
  beta_nu_mapping.reserve(dim);
  gamma_eta_mapping.reserve(dim);
  beta_gamma_nu_eta_mapping.reserve(dim);

  // partial derivatives table
  arma::mat derivative_map_beta;
  ka_->AllocateDerivativeMap(dim, order2, derivative_map_beta);
  arma::mat derivative_map_gamma;
  ka_->AllocateDerivativeMap(dim, order3, derivative_map_gamma);
  
  // compute center differences and complete the table of partial derivatives
  arma::vec xI_xK(dim), xJ_xK(dim);
  arma::vec& xJ_center = fe2.get_center();
  arma::vec& xK_center = fe3.get_center();

  for(index_t d = 0; d < dim; d++) {
    xI_xK[d] = (center_[d] - xK_center[d]) / bandwidth_factor;
    xJ_xK[d] = (xJ_center[d] - xK_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xK, derivative_map_beta, order2);
  ka_->ComputeDirectionalDerivatives(xJ_xK, derivative_map_gamma, order3);

  // inverse factorials
  arma::vec& inv_multiindex_factorials = sea_->get_inv_multiindex_factorials();

  // precompute pairwise kernel values between node i and node j
  arma::mat exhaustive_ij(node1_end - node1_begin, node2_end - node2_begin);
  for(index_t i = node1_begin; i < node1_end; i++) {
    arma::vec i_col = data.unsafe_col(i);
    for(index_t j = node2_begin; j < node2_end; j++) {
      arma::vec j_col = data.unsafe_col(j);
      
      exhaustive_ij(i - node1_begin, j - node2_begin) =
	 kernel_->EvalUnnormOnSq(la::DistanceSqEuclidean(i_col, j_col));
    }
  }

  // main loop
  for(index_t beta = 0; beta < total_num_coeffs2; beta++) {
    
    const std::vector<short int> &beta_mapping = multiindex_mapping[beta];
    const std::vector<short int> &lower_mappings_for_beta = 
      lower_mapping_index[beta];
    double beta_derivative = ka_->ComputePartialDerivative
      (derivative_map_beta, beta_mapping);
    
    for(index_t nu = 0; nu < lower_mappings_for_beta.size(); nu++) {
      
      const std::vector<short int> &nu_mapping = 
	multiindex_mapping[lower_mappings_for_beta[nu]];
      
      // beta - nu
      for(index_t d = 0; d < dim; d++) {
	beta_nu_mapping[d] = beta_mapping[d] - nu_mapping[d];
      }
      
      for(index_t gamma = 0; gamma < total_num_coeffs3; gamma++) {
	
	const std::vector<short int> &gamma_mapping = multiindex_mapping[gamma];
	const std::vector<short int> &lower_mappings_for_gamma = 
	  lower_mapping_index[gamma];
	double gamma_derivative = ka_->ComputePartialDerivative
	  (derivative_map_gamma, gamma_mapping);
	
	for(index_t eta = 0; eta < lower_mappings_for_gamma.size(); 
	    eta++) {
	  
	  // add up alpha, mu, eta and beta, gamma, nu, eta
	  int sign = 0;
	  
	  const std::vector<short int> &eta_mapping =
	    multiindex_mapping[lower_mappings_for_gamma[eta]];
	  
	  for(index_t d = 0; d < dim; d++) {
	    beta_gamma_nu_eta_mapping[d] = beta_mapping[d] +
	      gamma_mapping[d] - nu_mapping[d] - eta_mapping[d];
	    gamma_eta_mapping[d] = gamma_mapping[d] - eta_mapping[d];
	    
	    sign += 2 * (beta_mapping[d] + gamma_mapping[d]) - 
	      (nu_mapping[d] + eta_mapping[d]);
	  }
	  if(sign % 2 == 1) {
	    sign = -1;
	  }
	  else {
	    sign = 1;
	  }
	  
	  // retrieve moments for appropriate multiindex maps
	  moment_k = coeffs3[sea_->ComputeMultiindexPosition
			     (beta_gamma_nu_eta_mapping)];

	  // loop over every pairs of points in node i and node j
	  for(index_t i = node1_begin; i < node1_end; i++) {
	    
	    xi_xI = 
	      inv_multiindex_factorials
	      [sea_->ComputeMultiindexPosition(nu_mapping)];
	    for(index_t d = 0; d < dim; d++) {
	      diff = (data(d, i) - center_[d]) / bandwidth_factor;
	      xi_xI *= pow(diff, nu_mapping[d]);
	    }	    

	    for(index_t j = node2_begin; j < node2_end; j++) {

	      xj_xJ = inv_multiindex_factorials
		[sea_->ComputeMultiindexPosition(eta_mapping)];
	      for(index_t d = 0; d < dim; d++) {
		diff = (data(d, j) - xJ_center[d]) / bandwidth_factor;
		xj_xJ *= pow(diff, eta_mapping[d]);
	      }

	      result = sign *
		sea_->get_n_multichoose_k_by_pos
                (sea_->ComputeMultiindexPosition(beta_gamma_nu_eta_mapping),
                 sea_->ComputeMultiindexPosition(beta_nu_mapping)) *
		beta_derivative * gamma_derivative * xi_xI * xj_xJ *
		moment_k * exhaustive_ij(i - node1_begin, j - node2_begin);
	      
	      if(result > 0) {
		pos_sum += result;
	      }
	      else {
		neg_sum += result;
	      }
	    }
	  }
	  
	} // end of eta
      } // end of gamma
    } // end of nu
  } // end of beta
  
  // combine negative and positive sums
  sum = neg_sum + pos_sum;
  return sum;
}

template<typename TKernelAux>
double FarFieldExpansion<TKernelAux>::ConvolveField
(const FarFieldExpansion &fe, int order) const {
  
  // The bandwidth factor and the multiindex mapping stuffs.
  double bandwidth_factor = ka_->BandwidthFactor(bandwidth_sq());
  const std::vector<std::vector<short int> >& multiindex_mapping = 
    sea_->get_multiindex_mapping();
  const std::vector<std::vector<short int> >& lower_mapping_index = 
    sea_->get_lower_mapping_index();
  
  // Get the total number of coefficients and the coefficient themselves.
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  int dim = sea_->get_dimension();
  const arma::vec& coeffs2 = fe.get_coeffs();

  // Actual accumulated sum.
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;

  // The partial derivatives table.
  arma::mat derivative_map_alpha;
  ka_->AllocateDerivativeMap(dim, order, derivative_map_alpha);

  // Compute the center difference and its table of partial
  // derivatives.
  arma::vec xI_xJ(dim);
  const arma::vec& xJ_center = fe.get_center();

  for(index_t d = 0; d < dim; d++) {
    xI_xJ[d] = (center_[d] - xJ_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xJ, derivative_map_alpha, order);

  // The inverse factorials.
  const arma::vec& inv_multiindex_factorials = sea_->get_inv_multiindex_factorials();

  // The temporary space for computing the difference of two mappings.
  std::vector<short int> alpha_minus_beta_mapping;
  alpha_minus_beta_mapping.reserve(dim);

  // The main loop.
  for(index_t alpha = 0; alpha < total_num_coeffs; alpha++) {

    const std::vector<short int> &alpha_mapping = multiindex_mapping[alpha];
    const std::vector<short int> &lower_mappings_for_alpha = 
      lower_mapping_index[alpha];
    double alpha_derivative = ka_->ComputePartialDerivative
      (derivative_map_alpha, alpha_mapping);
    
    for(index_t beta = 0; beta < lower_mappings_for_alpha.size(); beta++) {
      
      const std::vector<short int> &beta_mapping = 
	multiindex_mapping[lower_mappings_for_alpha[beta]];
      
      double n_choose_k_factor = sea_->get_n_multichoose_k_by_pos
	(sea_->ComputeMultiindexPosition(alpha_mapping),
	 sea_->ComputeMultiindexPosition(beta_mapping));

      // Compute the sign-changes based on the multi-index map
      // difference.
      int map_difference = 0;

      for(index_t d = 0; d < dim; d++) {
	alpha_minus_beta_mapping[d] = alpha_mapping[d] - beta_mapping[d];
	map_difference += alpha_minus_beta_mapping[d];
      }
      
      // Current iteration's contribution to the sum.
      double contribution = alpha_derivative * n_choose_k_factor *
	coeffs_[sea_->ComputeMultiindexPosition(beta_mapping)] *
	coeffs2[sea_->ComputeMultiindexPosition(alpha_minus_beta_mapping)];

      // Flip the sign of the contribution if the map difference is an
      // odd number.
      if(map_difference % 2 == 1) {
	contribution = -contribution;
      }

      if(contribution < 0) {
	neg_sum += contribution;
      }
      else {
	pos_sum += contribution;
      }
    }    
  }
  sum = pos_sum + neg_sum;
  return sum;
}

template<typename TKernelAux>
double FarFieldExpansion<TKernelAux>::ConvolveField
(const FarFieldExpansion &fe2, const FarFieldExpansion &fe3,
 int order1, int order2, int order3) const {
  
  // bandwidth factor and multiindex mapping stuffs
  double result;
  double bandwidth_factor = ka_->BandwidthFactor(bandwidth_sq());
  const std::vector<short int>& multiindex_mapping = 
    sea_->get_multiindex_mapping();
  const std::vector<short int>& lower_mapping_index = 
    sea_->get_lower_mapping_index();

  // get the total number of coefficients and coefficients
  int total_num_coeffs1 = sea_->get_total_num_coeffs(order1);
  int total_num_coeffs2 = sea_->get_total_num_coeffs(order2);
  int total_num_coeffs3 = sea_->get_total_num_coeffs(order3);
  int dim = sea_->get_dimension();
  arma::vec& coeffs2 = fe2.get_coeffs();
  arma::vec& coeffs3 = fe3.get_coeffs();

  // actual accumulated sum
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;
  
  // some temporary
  double moment_i, moment_j, moment_k;

  // temporary array
  std::vector<short int> mu_nu_mapping;
  std::vector<short int> alpha_mu_eta_mapping;
  std::vector<short int> beta_gamma_nu_eta_mapping;
  std::vector<short int> alpha_mu_mapping;
  std::vector<short int> beta_nu_mapping;
  std::vector<short int> gamma_eta_mapping;
  alpha_mu_mapping.reserve(dim);
  beta_nu_mapping.reserve(dim);
  gamma_eta_mapping.reserve(dim);
  mu_nu_mapping.reserve(dim);
  alpha_mu_eta_mapping.reserve(dim);
  beta_gamma_nu_eta_mapping.reserve(dim);

  // partial derivatives table
  arma::mat derivative_map_alpha;
  ka_->AllocateDerivativeMap(dim, order1, derivative_map_alpha);
  arma::mat derivative_map_beta;
  ka_->AllocateDerivativeMap(dim, order2, derivative_map_beta);
  arma::mat derivative_map_gamma;
  ka_->AllocateDerivativeMap(dim, order3, derivative_map_gamma);
  
  // compute center differences and complete the table of partial derivatives
  arma::vec xI_xJ(dim), xI_xK(dim), xJ_xK(dim);
  arma::vec& xJ_center = fe2.get_center();
  arma::vec& xK_center = fe3.get_center();

  for(index_t d = 0; d < dim; d++) {
    xI_xJ[d] = (center_[d] - xJ_center[d]) / bandwidth_factor;
    xI_xK[d] = (center_[d] - xK_center[d]) / bandwidth_factor;
    xJ_xK[d] = (xJ_center[d] - xK_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xJ, derivative_map_alpha, order1);
  ka_->ComputeDirectionalDerivatives(xI_xK, derivative_map_beta, order2);
  ka_->ComputeDirectionalDerivatives(xJ_xK, derivative_map_gamma, order3);

  // inverse factorials
  arma::vec& inv_multiindex_factorials = sea_->get_inv_multiindex_factorials();

  // main loop
  for(index_t alpha = 0; alpha < total_num_coeffs1; alpha++) {

    const std::vector<short int>& alpha_mapping = multiindex_mapping[alpha];
    const std::vector<short int>& lower_mappings_for_alpha = 
      lower_mapping_index[alpha];
    double alpha_derivative = ka_->ComputePartialDerivative
      (derivative_map_alpha, alpha_mapping);

    for(index_t mu = 0; mu < lower_mappings_for_alpha.size(); mu++) {

      const std::vector<short int>& mu_mapping = 
	multiindex_mapping[lower_mappings_for_alpha[mu]];

      // alpha - mu
      for(index_t d = 0; d < dim; d++) {
	alpha_mu_mapping[d] = alpha_mapping[d] - mu_mapping[d];
      }
      
      for(index_t beta = 0; beta < total_num_coeffs2; beta++) {
	
	const std::vector<short int>& beta_mapping = multiindex_mapping[beta];
	const std::vector<short int>& lower_mappings_for_beta = 
	  lower_mapping_index[beta];
	double beta_derivative = ka_->ComputePartialDerivative
	  (derivative_map_beta, beta_mapping);

	for(index_t nu = 0; nu < lower_mappings_for_beta.size(); nu++) {
	  
	  const std::vector<short int> &nu_mapping = 
	    multiindex_mapping[lower_mappings_for_beta[nu]];

	  // mu + nu and beta - nu
	  for(index_t d = 0; d < dim; d++) {
	    mu_nu_mapping[d] = mu_mapping[d] + nu_mapping[d];
	    beta_nu_mapping[d] = beta_mapping[d] - nu_mapping[d];
	  }

	  for(index_t gamma = 0; gamma < total_num_coeffs3; gamma++) {
	    
	    const std::vector<short int> &gamma_mapping = 
	      multiindex_mapping[gamma];
	    const std::vector<short int> &lower_mappings_for_gamma = 
	      lower_mapping_index[gamma];
	    double gamma_derivative = ka_->ComputePartialDerivative
	      (derivative_map_gamma, gamma_mapping);

	    for(index_t eta = 0; eta < lower_mappings_for_gamma.size(); 
		eta++) {
	      
	      // add up alpha, mu, eta and beta, gamma, nu, eta
	      int sign = 0;
	      
	      const std::vector<short int>& eta_mapping =
		multiindex_mapping[lower_mappings_for_gamma[eta]];

	      for(index_t d = 0; d < dim; d++) {
		alpha_mu_eta_mapping[d] = alpha_mapping[d] - mu_mapping[d] +
		  eta_mapping[d];
		beta_gamma_nu_eta_mapping[d] = beta_mapping[d] +
		  gamma_mapping[d] - nu_mapping[d] - eta_mapping[d];
		gamma_eta_mapping[d] = gamma_mapping[d] - eta_mapping[d];
		
		sign += 2 * (alpha_mapping[d] + beta_mapping[d] + 
			     gamma_mapping[d]) - mu_mapping[d] - 
		  nu_mapping[d] - eta_mapping[d];
	      }
	      if(sign % 2 == 1) {
		sign = -1;
	      }
	      else {
		sign = 1;
	      }
	      
	      // retrieve moments for appropriate multiindex maps
	      moment_i = 
		coeffs_[sea_->ComputeMultiindexPosition(mu_nu_mapping)];
	      moment_j = 
		coeffs2[sea_->ComputeMultiindexPosition
			(alpha_mu_eta_mapping)];
	      moment_k = 
		coeffs3[sea_->ComputeMultiindexPosition
			(beta_gamma_nu_eta_mapping)];

	      result = sign * 
		sea_->get_n_multichoose_k_by_pos
		(sea_->ComputeMultiindexPosition(mu_nu_mapping),
		 sea_->ComputeMultiindexPosition(mu_mapping)) *
		sea_->get_n_multichoose_k_by_pos
		(sea_->ComputeMultiindexPosition(alpha_mu_eta_mapping),
		 sea_->ComputeMultiindexPosition(eta_mapping)) *
		sea_->get_n_multichoose_k_by_pos
		(sea_->ComputeMultiindexPosition(beta_gamma_nu_eta_mapping),
		 sea_->ComputeMultiindexPosition(beta_nu_mapping)) *
		alpha_derivative * beta_derivative * gamma_derivative * 
		moment_i * moment_j * moment_k;

	      if(result > 0) {
		pos_sum += result;
	      }
	      else {
		neg_sum += result;
	      }

	    } // end of eta
	  } // end of gamma
	} // end of nu
      } // end of beta
    } // end of mu
  } // end of alpha
  
  // combine negative and positive sums
  sum = neg_sum + pos_sum;
  return sum;
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::Init(const arma::vec& center, 
					 const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  center_ = center;
  order_ = -1;
  sea_ = &(ka.sea_);
  ka_ = &ka;

  // initialize coefficient array
  coeffs_.zeros(sea_->get_max_total_num_coeffs());
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::Init(const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);  
  order_ = -1;
  sea_ = &(ka.sea_);
  center_.zeros(sea_->get_dimension());
  ka_ = &ka;

  // initialize coefficient array
  coeffs_.zeros(sea_->get_max_total_num_coeffs());
}

template<typename TKernelAux>
template<typename TBound>
int FarFieldExpansion<TKernelAux>::OrderForConvolving
(const TBound &far_field_region, const arma::vec& far_field_region_centroid,
 const TBound &local_field_region, const arma::vec& local_field_region_centroid,
 double min_dist_sqd_regions, double max_dist_sqd_regions, double max_error,
 double *actual_error) const {
  
  return ka_->OrderForConvolvingFarField
    (far_field_region, far_field_region_centroid, local_field_region,
     local_field_region_centroid, min_dist_sqd_regions, max_dist_sqd_regions,
     max_error, actual_error);
}

template<typename TKernelAux>
template<typename TBound>
int FarFieldExpansion<TKernelAux>::OrderForEvaluating
(const TBound &far_field_region, 
 const TBound &local_field_region, double min_dist_sqd_regions,
 double max_dist_sqd_regions, double max_error, double *actual_error) const {
  
  return ka_->OrderForEvaluatingFarField(far_field_region,
					 local_field_region,
					 min_dist_sqd_regions, 
					 max_dist_sqd_regions, max_error,
					 actual_error);
}

template<typename TKernelAux>
template<typename TBound>
int FarFieldExpansion<TKernelAux>::
OrderForConvertingToLocal(const TBound &far_field_region,
			  const TBound &local_field_region, 
			  double min_dist_sqd_regions, 
			  double max_dist_sqd_regions,
			  double max_error, 
			  double *actual_error) const {
  
  return ka_->OrderForConvertingFromFarFieldToLocal(far_field_region,
						    local_field_region,
						    min_dist_sqd_regions,
						    max_dist_sqd_regions,
						    max_error, actual_error);
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::PrintDebug(const char *name, 
					       FILE *stream) const {
  
  int dim = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Far field expansion\n");
  fprintf(stream, "Center: ");
  
  for (index_t i = 0; i < center_.n_elem; i++) {
    fprintf(stream, "%g ", center_[i]);
  }
  fprintf(stream, "\n");
  
  fprintf(stream, "f(");
  for(index_t d = 0; d < dim; d++) {
    fprintf(stream, "x_q%d", d);
    if(d < dim - 1)
      fprintf(stream, ",");
  }
  fprintf(stream, ") = \\sum\\limits_{x_r \\in R} K(||x_q - x_r||) = ");
  
  for (index_t i = 0; i < total_num_coeffs; i++) {
    const std::vector<short int> &mapping = sea_->get_multiindex(i);
    fprintf(stream, "%g ", coeffs_[i]);
    
    fprintf(stream, "(-1)^(");
    for(index_t d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);
      if(d < dim - 1)
	fprintf(stream, " + ");
    }
    fprintf(stream, ") D^((");
    for(index_t d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);
      
      if(d < dim - 1)
	fprintf(stream, ",");
    }
    fprintf(stream, ")) f(x_q - x_R)");
    if(i < total_num_coeffs - 1) {
      fprintf(stream, " + ");
    }
  }
  fprintf(stream, "\n");
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::TranslateFromFarField
(const FarFieldExpansion &se) {
  
  double bandwidth_factor = ka_->BandwidthFactor(se.bandwidth_sq());
  int dim = sea_->get_dimension();
  int order = se.get_order();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  const arma::vec& prev_coeffs = se.get_coeffs();
  const arma::vec& prev_center = se.get_center();
  const std::vector<std::vector<short int> >& multiindex_mapping = 
    sea_->get_multiindex_mapping();
  const std::vector<std::vector<short int> >& lower_mapping_index = 
    sea_->get_lower_mapping_index();

  std::vector<short int> tmp_storage;
  arma::vec center_diff(dim);
  const arma::vec& inv_multiindex_factorials =
      sea_->get_inv_multiindex_factorials();

  // retrieve coefficients to be translated and helper mappings
  tmp_storage.reserve(sea_->get_dimension());

  // no coefficients can be translated
  if(order == -1)
    return;
  else
    order_ = order;
  
  // compute center difference
  for(index_t j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  for(index_t j = 0; j < total_num_coeffs; j++) {
    
    const std::vector<short int>& gamma_mapping = multiindex_mapping[j];
    const std::vector<short int>& lower_mappings_for_gamma = 
      lower_mapping_index[j];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(index_t k = 0; k < lower_mappings_for_gamma.size(); k++) {

      const std::vector<short int>& inner_mapping = 
	multiindex_mapping[lower_mappings_for_gamma[k]];

      int flag = 0;
      double diff1;
      
      // compute gamma minus alpha
      for(index_t l = 0; l < dim; l++) {
	tmp_storage[l] = gamma_mapping[l] - inner_mapping[l];

	if(tmp_storage[l] < 0) {
	  flag = 1;
	  break;
	}
      }
      
      if(flag) {
	continue;
      }
      
      diff1 = 1.0;
      
      for(index_t l = 0; l < dim; l++) {
	diff1 *= pow(center_diff[l] / bandwidth_factor, tmp_storage[l]);
      }

      double prod = prev_coeffs[lower_mappings_for_gamma[k]] * diff1 * 
	inv_multiindex_factorials
	[sea_->ComputeMultiindexPosition(tmp_storage)];
      
      if(prod > 0) {
	pos_coeff += prod;
      }
      else {
	neg_coeff += prod;
      }

    } // end of k-loop
    
    coeffs_[j] += pos_coeff + neg_coeff;

  } // end of j-loop
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::TranslateToLocal
(LocalExpansion<TKernelAux> &se, int truncation_order) {
  
  arma::vec pos_arrtmp, neg_arrtmp;
  arma::mat derivative_map;
  ka_->AllocateDerivativeMap(sea_->get_dimension(), 2 * truncation_order,
			     derivative_map);
  const arma::vec& local_center = se.get_center();
  arma::vec cent_diff;
  arma::vec& local_coeffs = se.get_coeffs();
  int local_order = se.get_order();
  int dimension = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(truncation_order);
  double bandwidth_factor = ka_->BandwidthFactor(se.bandwidth_sq());

  // get center and coefficients for local expansion
  cent_diff.set_size(dimension);

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(local_order < truncation_order) {
    se.set_order(truncation_order);
  }

  // Compute derivatives.
  pos_arrtmp.set_size(total_num_coeffs);
  neg_arrtmp.set_size(total_num_coeffs);

  // Compute center difference divided by the bandwidth factor.
  for(index_t j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // Compute required partial derivatives.
  ka_->ComputeDirectionalDerivatives(cent_diff, derivative_map,
				     2 * truncation_order);
  std::vector<short int> beta_plus_alpha;
  beta_plus_alpha.reserve(dimension);

  for(index_t j = 0; j < total_num_coeffs; j++) {

    const std::vector<short int>& beta_mapping = sea_->get_multiindex(j);
    pos_arrtmp[j] = neg_arrtmp[j] = 0;

    for(index_t k = 0; k < total_num_coeffs; k++) {

      const std::vector<short int>& alpha_mapping = sea_->get_multiindex(k);
      for(index_t d = 0; d < dimension; d++) {
	beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
	ka_->ComputePartialDerivative(derivative_map, beta_plus_alpha);
      
      double prod = coeffs_[k] * derivative_factor;

      if(prod > 0) {
	pos_arrtmp[j] += prod;
      }
      else {
	neg_arrtmp[j] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  arma::vec C_k_neg = sea_->get_neg_inv_multiindex_factorials();
  for(index_t j = 0; j < total_num_coeffs; j++) {
    local_coeffs[j] += (pos_arrtmp[j] + neg_arrtmp[j]) * C_k_neg[j];
  }
}

#endif
