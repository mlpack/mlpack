#ifndef INSIDE_MULT_FARFIELD_EXPANSION_H
#error "This file is not a public header file!"
#endif

#ifndef MULT_FARFIELD_EXPANSION_IMPL_H
#define MULT_FARFIELD_EXPANSION_IMPL_H

template<typename TKernelAux>
void MultFarFieldExpansion<TKernelAux>::AccumulateCoeffs(const Matrix& data, 
							 const Vector& weights,
							 int begin, int end, 
							 int order) {
  
  int dim = data.n_rows();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  int max_total_num_coeffs = sea_->get_max_total_num_coeffs();
  Vector x_r, tmp;
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  x_r.Init(dim);
  tmp.Init(max_total_num_coeffs);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(max_total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(max_total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  
  // get the order of traversal for the given order of approximation
  const ArrayList<int> &traversal_order = sea_->traversal_mapping_[order_];

  // Repeat for each reference point in this reference node.
  for(index_t r = begin; r < end; r++) {

    // Calculate the coordinate difference between the ref point and the 
    // centroid.
    for(index_t i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
    }
    
    tmp.SetZero();
    tmp[0] = 1.0;

    for(index_t i = 1; i < total_num_coeffs; i++) {
      
      int index = traversal_order[i];
      const ArrayList<int> &lower_mappings = sea_->lower_mapping_index_[index];

      // from the direct descendant, recursively compute the multipole moments
      int direct_ancestor_mapping_pos = 
	lower_mappings[lower_mappings.size() - 2];

      int position = 0;
      const ArrayList<int> &mapping = sea_->multiindex_mapping_[index];
      const ArrayList<int> &direct_ancestor_mapping = 
	sea_->multiindex_mapping_[direct_ancestor_mapping_pos];
      for(index_t i = 0; i < dim; i++) {
	if(mapping[i] != direct_ancestor_mapping[i]) {
	  position = i;
	  break;
	}
      }
      
      tmp[index] = tmp[direct_ancestor_mapping_pos] * x_r[position];
    }

    // Tally up the result in A_k.
    for(index_t i = 0; i < total_num_coeffs; i++) {

      int index = traversal_order[i];
      double prod = weights[r] * tmp[index];
      
      if(prod > 0) {
	pos_coeffs[index] += prod;
      }
      else {
	neg_coeffs[index] += prod;
      }
    }

  } // End of looping through each reference point

  for(index_t r = 0; r < total_num_coeffs; r++) {
    int index = traversal_order[r];
    coeffs_[index] += (pos_coeffs[index] + neg_coeffs[index]) * 
      sea_->inv_multiindex_factorials_[index];
  }
}

template<typename TKernelAux>
double MultFarFieldExpansion<TKernelAux>::ConvolveField
(const MultFarFieldExpansion &fe, int order) const {
  
  // The bandwidth factor and the multiindex mapping stuffs.
  double bandwidth_factor = ka_->BandwidthFactor(bandwidth_sq());
  const ArrayList<int> *multiindex_mapping = sea_->get_multiindex_mapping();
  const ArrayList<int> *lower_mapping_index = sea_->get_lower_mapping_index();
  
  // Get the total number of coefficients and the coefficient themselves.
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  int dim = sea_->get_dimension();
  Vector coeffs2;
  coeffs2.Alias(fe.get_coeffs());

  // Actual accumulated sum.
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;

  // The partial derivatives table.
  Matrix derivative_map_alpha;
  derivative_map_alpha.Init(dim, order + 1);

  // Compute the center difference and its table of partial
  // derivatives.
  Vector xI_xJ;
  xI_xJ.Init(dim);
  Vector xJ_center;
  xJ_center.Alias(*(fe.get_center()));

  for(index_t d = 0; d < dim; d++) {
    xI_xJ[d] = (center_[d] - xJ_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xJ, derivative_map_alpha);

  // The inverse factorials.
  Vector inv_multiindex_factorials;
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // The temporary space for computing the difference of two mappings.
  ArrayList<index_t> alpha_minus_beta_mapping;
  alpha_minus_beta_mapping.Init(dim);

  // The main loop.
  for(index_t alpha = 0; alpha < total_num_coeffs; alpha++) {

    const ArrayList <int> &alpha_mapping = multiindex_mapping[alpha];
    const ArrayList <int> &lower_mappings_for_alpha = 
      lower_mapping_index[alpha];
    double alpha_derivative = ka_->ComputePartialDerivative
      (derivative_map_alpha, alpha_mapping);
    
    for(index_t beta = 0; beta < lower_mappings_for_alpha.size(); beta++) {
      
      const ArrayList <int> &beta_mapping = 
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
void MultFarFieldExpansion<TKernelAux>::RefineCoeffs(const Matrix& data, 
						     const Vector& weights, 
						     int begin, int end, 
						     int order) {

  // if we already have the order of approximation, then return.
  if(order_ >= order) {
    return;
  }

  // otherwise, recompute from scratch... this could be improved potentially
  // but I believe it will not squeeze out more performance (as in O(D^p)
  // expansions).
  else {
    order_ = order;
    
    coeffs_.SetZero();
    AccumulateCoeffs(data, weights, begin, end, order);
  }
}

template<typename TKernelAux>
double MultFarFieldExpansion<TKernelAux>::EvaluateField(const Matrix& data, 
							int row_num, 
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
  Matrix derivative_map;
  derivative_map.Init(dim, order_ + 1);

  // temporary variable
  Vector arrtmp;
  arrtmp.Init(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  Vector x_q_minus_x_R;
  x_q_minus_x_R.Init(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(index_t d = 0; d < dim; d++) {
    x_q_minus_x_R[d] = (data.get(d, row_num) - center_[d]) / bandwidth_factor;
  }

  // compute deriative maps based on coordinate difference.
  ka_->ComputeDirectionalDerivatives(x_q_minus_x_R, derivative_map);
  
  // get the order of traversal for the given order of approximation
  const ArrayList<int> &traversal_order = sea_->traversal_mapping_[order_];

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(index_t j = 0; j < total_num_coeffs; j++) {
    
    int index = traversal_order[j];
    const ArrayList<int> &mapping = sea_->get_multiindex(index);
    double arrtmp = ka_->ComputePartialDerivative(derivative_map, mapping);
    double prod = coeffs_[index] * arrtmp;
    
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
double MultFarFieldExpansion<TKernelAux>::EvaluateField(const Vector& x_q, 
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
  Matrix derivative_map;
  derivative_map.Init(dim, order_ + 1);

  // temporary variable
  Vector arrtmp;
  arrtmp.Init(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  Vector x_q_minus_x_R;
  x_q_minus_x_R.Init(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(index_t d = 0; d < dim; d++) {
    x_q_minus_x_R[d] = (x_q[d] - center_[d]) / bandwidth_factor;
  }

  // compute deriative maps based on coordinate difference.
  ka_->ComputeDirectionalDerivatives(x_q_minus_x_R, derivative_map);
  
  // get the order of traversal for the given order of approximation
  ArrayList<int> traversal_order = sea_->traversal_mapping_[order_];

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(index_t j = 0; j < total_num_coeffs; j++) {
    
    int index = traversal_order[j];
    ArrayList<int> mapping = sea_->get_multiindex(index);
    double arrtmp = ka_->ComputePartialDerivative(derivative_map, mapping);
    double prod = coeffs_[index] * arrtmp;
    
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
void MultFarFieldExpansion<TKernelAux>::Init(const Vector& center, 
					     const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  center_.Copy(center);
  order_ = -1;
  sea_ = &(ka.sea_);
  ka_ = &ka;

  // Initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernelAux>
  void MultFarFieldExpansion<TKernelAux>::Init(const TKernelAux &ka) {
  
  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);  
  order_ = -1;
  sea_ = &(ka.sea_);
  center_.Init(sea_->get_dimension());
  center_.SetZero();
  ka_ = &ka;

  // Initialize coefficient array.
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}


template<typename TKernelAux>
template<typename TBound>
int MultFarFieldExpansion<TKernelAux>::OrderForEvaluating
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
int MultFarFieldExpansion<TKernelAux>::
OrderForConvertingToLocal(const TBound &far_field_region,
			  const TBound &local_field_region, 
			  double min_dist_sqd_regions, 
			  double max_dist_sqd_regions,
			  double max_error, 
			  double *actual_error) const {
  
  return ka_->OrderForConvertingFromFarFieldToLocal
    (far_field_region, local_field_region, min_dist_sqd_regions,
     max_dist_sqd_regions, max_error, actual_error);
}

template<typename TKernelAux>
void MultFarFieldExpansion<TKernelAux>::PrintDebug
(const char *name, FILE *stream) const {
    
  int dim = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Far field expansion\n");
  fprintf(stream, "Center: ");
  
  for (index_t i = 0; i < center_.length(); i++) {
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
    const ArrayList<int> &mapping = sea_->get_multiindex(i);
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
void MultFarFieldExpansion<TKernelAux>::TranslateFromFarField
(const MultFarFieldExpansion &se) {
  
  double bandwidth_factor = ka_->BandwidthFactor(se.bandwidth_sq());
  int dim = sea_->get_dimension();
  int order = se.get_order();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector prev_coeffs;
  Vector prev_center;
  const ArrayList < int > *multiindex_mapping = sea_->get_multiindex_mapping();
  const ArrayList < int > *lower_mapping_index = 
    sea_->get_lower_mapping_index();

  ArrayList <int> tmp_storage;
  Vector center_diff;
  Vector inv_multiindex_factorials;

  center_diff.Init(dim);

  // retrieve coefficients to be translated and helper mappings
  prev_coeffs.Alias(se.get_coeffs());
  prev_center.Alias(*(se.get_center()));
  tmp_storage.Init(sea_->get_dimension());
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // no coefficients can be translated
  if(order == -1) {
    return;
  }
  else {
    order_ = order;
  }
  
  // compute center difference
  for(index_t j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  // get the order of traversal for the given order of approximation
  const ArrayList<int> &traversal_order = sea_->traversal_mapping_[order];

  for(index_t j = 0; j < total_num_coeffs; j++) {
   
    int index = traversal_order[j];
    const ArrayList <int> &gamma_mapping = multiindex_mapping[index];
    const ArrayList <int> &lower_mappings_for_gamma = 
      lower_mapping_index[index];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(index_t k = 0; k < lower_mappings_for_gamma.size(); k++) {

      const ArrayList <int> &inner_mapping = 
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
void MultFarFieldExpansion<TKernelAux>::TranslateToLocal
(MultLocalExpansion<TKernelAux> &se, int truncation_order) {
  
  Vector pos_arrtmp, neg_arrtmp;
  Matrix derivative_map;
  Vector local_center;
  Vector cent_diff;
  Vector local_coeffs;
  int local_order = se.get_order();
  int dimension = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(truncation_order);
  int limit;
  double bandwidth_factor = ka_->BandwidthFactor(se.bandwidth_sq());

  // get center and coefficients for local expansion
  local_center.Alias(*(se.get_center()));
  local_coeffs.Alias(se.get_coeffs());
  cent_diff.Init(dimension);

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(local_order < truncation_order) {
    se.set_order(truncation_order);
  }

  // compute Gaussian derivative
  limit = 2 * truncation_order + 1;
  derivative_map.Init(dimension, limit);
  pos_arrtmp.Init(sea_->get_max_total_num_coeffs());
  neg_arrtmp.Init(sea_->get_max_total_num_coeffs());

  // compute center difference divided by bw_times_sqrt_two;
  for(index_t j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // compute required partial derivatives
  ka_->ComputeDirectionalDerivatives(cent_diff, derivative_map);
  ArrayList<int> beta_plus_alpha;
  beta_plus_alpha.Init(dimension);

  // get the order of traversal for the given order of approximation
  const ArrayList<int> &traversal_order = 
    sea_->traversal_mapping_[truncation_order];

  for(index_t j = 0; j < total_num_coeffs; j++) {

    int index = traversal_order[j];
    const ArrayList<int> &beta_mapping = sea_->get_multiindex(index);
    pos_arrtmp[index] = neg_arrtmp[index] = 0;

    for(index_t k = 0; k < total_num_coeffs; k++) {

      int index_k = traversal_order[k];

      const ArrayList<int> &alpha_mapping = sea_->get_multiindex(index_k);
      for(index_t d = 0; d < dimension; d++) {
	beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
	ka_->ComputePartialDerivative(derivative_map, beta_plus_alpha);
      
      double prod = coeffs_[index_k] * derivative_factor;

      if(prod > 0) {
	pos_arrtmp[index] += prod;
      }
      else {
	neg_arrtmp[index] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  Vector C_k_neg = sea_->get_neg_inv_multiindex_factorials();
  for(index_t j = 0; j < total_num_coeffs; j++) {
    int index = traversal_order[j];
    local_coeffs[index] += (pos_arrtmp[index] + neg_arrtmp[index]) * 
      C_k_neg[index];
  }
}

#endif
