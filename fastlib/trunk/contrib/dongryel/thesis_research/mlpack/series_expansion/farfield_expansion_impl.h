#ifndef INSIDE_FARFIELD_EXPANSION_H
#error "This file is not a public header file!"
#endif

#ifndef FARFIELD_EXPANSION_IMPL_H
#define FARFIELD_EXPANSION_IMPL_H

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::Accumulate(const Vector &v, double weight,
    int order) {

  int dim = v.length();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int r, i, j, k, t, tail;
  GenVector<short int> heads;
  Vector x_r;
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  heads.Init(dim + 1);
  x_r.Init(dim);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  Vector C_k;

  // Calculate the coordinate difference between the ref point and the
  // centroid.
  for(i = 0; i < dim; i++) {
    x_r[i] = (v[i] - center_[i]) / bandwidth_factor;
  }

  // initialize heads
  heads.SetZero();
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
  C_k.Alias(sea_->get_inv_multiindex_factorials());

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::AccumulateCoeffs(const Matrix& data,
    const Vector& weights,
    int begin, int end,
    int order) {

  int dim = data.n_rows();
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  Vector tmp;
  int r, i, j, k, t, tail;
  GenVector<short int> heads;
  Vector x_r;
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  heads.Init(dim + 1);
  x_r.Init(dim);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  Vector C_k;

  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {

    // Calculate the coordinate difference between the ref point and the
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
    }

    // initialize heads
    heads.SetZero();
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
  C_k.Alias(sea_->get_inv_multiindex_factorials());

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::RefineCoeffs(const Matrix& data,
    const Vector& weights,
    int begin, int end,
    int order) {

  if(order_ < 0) {

    AccumulateCoeffs(data, weights, begin, end, order);
    return;
  }

  int dim = data.n_rows();
  int old_total_num_coeffs = sea_->get_total_num_coeffs(order_);
  int total_num_coeffs = sea_->get_total_num_coeffs(order);
  double tmp;
  int r, i, j;
  Vector x_r;
  double bandwidth_factor = ka_->BandwidthFactor(kernel_->bandwidth_sq());

  // initialize temporary variables
  x_r.Init(dim);
  Vector pos_coeffs;
  Vector neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // if we already have the order of approximation, then return.
  if(order_ >= order) {
    return;
  }
  else {
    order_ = order;
  }

  Vector C_k;

  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {

    // Calculate the coordinate difference between the ref point and the
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
    }

    // compute in bruteforce way
    for(i = old_total_num_coeffs; i < total_num_coeffs; i++) {
      const ArrayList<short int> &mapping = sea_->get_multiindex(i);
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

  // get multiindex factors
  C_k.Alias(sea_->get_inv_multiindex_factorials());

  for(r = old_total_num_coeffs; r < total_num_coeffs; r++) {
    coeffs_[r] = (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<typename TKernelAux>
double FarFieldExpansion<TKernelAux>::EvaluateField(const Matrix& data,
    int row_num,
    int order) const {
  return EvaluateField(data.GetColumnPtr(row_num), order);
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
  Matrix derivative_map;
  ka_->AllocateDerivativeMap(dim, order, &derivative_map);

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
  ka_->ComputeDirectionalDerivatives(x_q_minus_x_R, &derivative_map, order);

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(index_t j = 0; j < total_num_coeffs; j++) {
    const ArrayList<short int> &mapping = sea_->get_multiindex(j);
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
double FarFieldExpansion<TKernelAux>::MixField(const Matrix &data,
    int node1_begin, int node1_end,
    int node2_begin, int node2_end,
    const FarFieldExpansion &fe2,
    const FarFieldExpansion &fe3,
    int order2, int order3) const {

  // bandwidth factor and multiindex mapping stuffs
  double result;
  double bandwidth_factor = ka_->BandwidthFactor(bandwidth_sq());
  const ArrayList<short int> *multiindex_mapping =
    sea_->get_multiindex_mapping();
  const ArrayList<short int> *lower_mapping_index =
    sea_->get_lower_mapping_index();

  // get the total number of coefficients and coefficients
  int total_num_coeffs2 = sea_->get_total_num_coeffs(order2);
  int total_num_coeffs3 = sea_->get_total_num_coeffs(order3);
  int dim = sea_->get_dimension();
  Vector coeffs2, coeffs3;
  coeffs2.Alias(fe2.get_coeffs());
  coeffs3.Alias(fe3.get_coeffs());

  // actual accumulated sum
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;

  // some temporary
  double moment_k;
  double xi_xI, xj_xJ, diff;

  // temporary array
  ArrayList<short int> beta_gamma_nu_eta_mapping;
  ArrayList<short int> beta_nu_mapping;
  ArrayList<short int> gamma_eta_mapping;
  beta_nu_mapping.Init(dim);
  gamma_eta_mapping.Init(dim);
  beta_gamma_nu_eta_mapping.Init(dim);

  // partial derivatives table
  Matrix derivative_map_beta;
  ka_->AllocateDerivativeMap(dim, order2, &derivative_map_beta);
  Matrix derivative_map_gamma;
  ka_->AllocateDerivativeMap(dim, order3, &derivative_map_gamma);

  // compute center differences and complete the table of partial derivatives
  Vector xI_xK, xJ_xK;
  xI_xK.Init(dim);
  xJ_xK.Init(dim);
  Vector xJ_center, xK_center;
  xJ_center.Alias(*(fe2.get_center()));
  xK_center.Alias(*(fe3.get_center()));

  for(index_t d = 0; d < dim; d++) {
    xI_xK[d] = (center_[d] - xK_center[d]) / bandwidth_factor;
    xJ_xK[d] = (xJ_center[d] - xK_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xK, &derivative_map_beta, order2);
  ka_->ComputeDirectionalDerivatives(xJ_xK, &derivative_map_gamma, order3);

  // inverse factorials
  Vector inv_multiindex_factorials;
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // precompute pairwise kernel values between node i and node j
  Matrix exhaustive_ij;
  exhaustive_ij.Init(node1_end - node1_begin, node2_end - node2_begin);
  for(index_t i = node1_begin; i < node1_end; i++) {
    const double *i_col = data.GetColumnPtr(i);
    for(index_t j = node2_begin; j < node2_end; j++) {
      const double *j_col = data.GetColumnPtr(j);

      exhaustive_ij.set
      (i - node1_begin, j - node2_begin,
       kernel_->EvalUnnormOnSq(la::DistanceSqEuclidean(data.n_rows(),
                               i_col, j_col)));
    }
  }

  // main loop
  for(index_t beta = 0; beta < total_num_coeffs2; beta++) {

    const ArrayList <short int> &beta_mapping = multiindex_mapping[beta];
    const ArrayList <short int> &lower_mappings_for_beta =
      lower_mapping_index[beta];
    double beta_derivative = ka_->ComputePartialDerivative
                             (derivative_map_beta, beta_mapping);

    for(index_t nu = 0; nu < lower_mappings_for_beta.size(); nu++) {

      const ArrayList<short int> &nu_mapping =
        multiindex_mapping[lower_mappings_for_beta[nu]];

      // beta - nu
      for(index_t d = 0; d < dim; d++) {
        beta_nu_mapping[d] = beta_mapping[d] - nu_mapping[d];
      }

      for(index_t gamma = 0; gamma < total_num_coeffs3; gamma++) {

        const ArrayList <short int> &gamma_mapping = multiindex_mapping[gamma];
        const ArrayList <short int> &lower_mappings_for_gamma =
          lower_mapping_index[gamma];
        double gamma_derivative = ka_->ComputePartialDerivative
                                  (derivative_map_gamma, gamma_mapping);

        for(index_t eta = 0; eta < lower_mappings_for_gamma.size();
            eta++) {

          // add up alpha, mu, eta and beta, gamma, nu, eta
          int sign = 0;

          const ArrayList<short int> &eta_mapping =
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
              diff = (data.get(d, i) - center_[d]) / bandwidth_factor;
              xi_xI *= pow(diff, nu_mapping[d]);
            }

            for(index_t j = node2_begin; j < node2_end; j++) {

              xj_xJ = inv_multiindex_factorials
                      [sea_->ComputeMultiindexPosition(eta_mapping)];
              for(index_t d = 0; d < dim; d++) {
                diff = (data.get(d, j) - xJ_center[d]) / bandwidth_factor;
                xj_xJ *= pow(diff, eta_mapping[d]);
              }

              result = sign *
                       sea_->get_n_multichoose_k_by_pos
                       (sea_->ComputeMultiindexPosition(beta_gamma_nu_eta_mapping),
                        sea_->ComputeMultiindexPosition(beta_nu_mapping)) *
                       beta_derivative * gamma_derivative * xi_xI * xj_xJ *
                       moment_k * exhaustive_ij.get
                       (i - node1_begin, j - node2_begin);

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
  const ArrayList<short int> *multiindex_mapping =
    sea_->get_multiindex_mapping();
  const ArrayList<short int> *lower_mapping_index =
    sea_->get_lower_mapping_index();

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
  ka_->AllocateDerivativeMap(dim, order, &derivative_map_alpha);

  // Compute the center difference and its table of partial
  // derivatives.
  Vector xI_xJ;
  xI_xJ.Init(dim);
  Vector xJ_center;
  xJ_center.Alias(*(fe.get_center()));

  for(index_t d = 0; d < dim; d++) {
    xI_xJ[d] = (center_[d] - xJ_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xJ, &derivative_map_alpha, order);

  // The inverse factorials.
  Vector inv_multiindex_factorials;
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // The temporary space for computing the difference of two mappings.
  ArrayList<short int> alpha_minus_beta_mapping;
  alpha_minus_beta_mapping.Init(dim);

  // The main loop.
  for(index_t alpha = 0; alpha < total_num_coeffs; alpha++) {

    const ArrayList <short int> &alpha_mapping = multiindex_mapping[alpha];
    const ArrayList <short int> &lower_mappings_for_alpha =
      lower_mapping_index[alpha];
    double alpha_derivative = ka_->ComputePartialDerivative
                              (derivative_map_alpha, alpha_mapping);

    for(index_t beta = 0; beta < lower_mappings_for_alpha.size(); beta++) {

      const ArrayList <short int> &beta_mapping =
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
  const ArrayList<short int> *multiindex_mapping =
    sea_->get_multiindex_mapping();
  const ArrayList<short int> *lower_mapping_index =
    sea_->get_lower_mapping_index();

  // get the total number of coefficients and coefficients
  int total_num_coeffs1 = sea_->get_total_num_coeffs(order1);
  int total_num_coeffs2 = sea_->get_total_num_coeffs(order2);
  int total_num_coeffs3 = sea_->get_total_num_coeffs(order3);
  int dim = sea_->get_dimension();
  Vector coeffs2, coeffs3;
  coeffs2.Alias(fe2.get_coeffs());
  coeffs3.Alias(fe3.get_coeffs());

  // actual accumulated sum
  double neg_sum = 0;
  double pos_sum = 0;
  double sum = 0;

  // some temporary
  double moment_i, moment_j, moment_k;

  // temporary array
  ArrayList<short int> mu_nu_mapping;
  ArrayList<short int> alpha_mu_eta_mapping;
  ArrayList<short int> beta_gamma_nu_eta_mapping;
  ArrayList<short int> alpha_mu_mapping;
  ArrayList<short int> beta_nu_mapping;
  ArrayList<short int> gamma_eta_mapping;
  alpha_mu_mapping.Init(dim);
  beta_nu_mapping.Init(dim);
  gamma_eta_mapping.Init(dim);
  mu_nu_mapping.Init(dim);
  alpha_mu_eta_mapping.Init(dim);
  beta_gamma_nu_eta_mapping.Init(dim);

  // partial derivatives table
  Matrix derivative_map_alpha;
  ka_->AllocateDerivativeMap(dim, order1, &derivative_map_alpha);
  Matrix derivative_map_beta;
  ka_->AllocateDerivativeMap(dim, order2, &derivative_map_beta);
  Matrix derivative_map_gamma;
  ka_->AllocateDerivativeMap(dim, order3, &derivative_map_gamma);

  // compute center differences and complete the table of partial derivatives
  Vector xI_xJ, xI_xK, xJ_xK;
  xI_xJ.Init(dim);
  xI_xK.Init(dim);
  xJ_xK.Init(dim);
  Vector xJ_center, xK_center;
  xJ_center.Alias(*(fe2.get_center()));
  xK_center.Alias(*(fe3.get_center()));

  for(index_t d = 0; d < dim; d++) {
    xI_xJ[d] = (center_[d] - xJ_center[d]) / bandwidth_factor;
    xI_xK[d] = (center_[d] - xK_center[d]) / bandwidth_factor;
    xJ_xK[d] = (xJ_center[d] - xK_center[d]) / bandwidth_factor;
  }
  ka_->ComputeDirectionalDerivatives(xI_xJ, &derivative_map_alpha, order1);
  ka_->ComputeDirectionalDerivatives(xI_xK, &derivative_map_beta, order2);
  ka_->ComputeDirectionalDerivatives(xJ_xK, &derivative_map_gamma, order3);

  // inverse factorials
  Vector inv_multiindex_factorials;
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

  // main loop
  for(index_t alpha = 0; alpha < total_num_coeffs1; alpha++) {

    const ArrayList <short int> &alpha_mapping = multiindex_mapping[alpha];
    const ArrayList <short int> &lower_mappings_for_alpha =
      lower_mapping_index[alpha];
    double alpha_derivative = ka_->ComputePartialDerivative
                              (derivative_map_alpha, alpha_mapping);

    for(index_t mu = 0; mu < lower_mappings_for_alpha.size(); mu++) {

      const ArrayList <short int> &mu_mapping =
        multiindex_mapping[lower_mappings_for_alpha[mu]];

      // alpha - mu
      for(index_t d = 0; d < dim; d++) {
        alpha_mu_mapping[d] = alpha_mapping[d] - mu_mapping[d];
      }

      for(index_t beta = 0; beta < total_num_coeffs2; beta++) {

        const ArrayList <short int> &beta_mapping = multiindex_mapping[beta];
        const ArrayList <short int> &lower_mappings_for_beta =
          lower_mapping_index[beta];
        double beta_derivative = ka_->ComputePartialDerivative
                                 (derivative_map_beta, beta_mapping);

        for(index_t nu = 0; nu < lower_mappings_for_beta.size(); nu++) {

          const ArrayList<short int> &nu_mapping =
            multiindex_mapping[lower_mappings_for_beta[nu]];

          // mu + nu and beta - nu
          for(index_t d = 0; d < dim; d++) {
            mu_nu_mapping[d] = mu_mapping[d] + nu_mapping[d];
            beta_nu_mapping[d] = beta_mapping[d] - nu_mapping[d];
          }

          for(index_t gamma = 0; gamma < total_num_coeffs3; gamma++) {

            const ArrayList <short int> &gamma_mapping =
              multiindex_mapping[gamma];
            const ArrayList <short int> &lower_mappings_for_gamma =
              lower_mapping_index[gamma];
            double gamma_derivative = ka_->ComputePartialDerivative
                                      (derivative_map_gamma, gamma_mapping);

            for(index_t eta = 0; eta < lower_mappings_for_gamma.size();
                eta++) {

              // add up alpha, mu, eta and beta, gamma, nu, eta
              int sign = 0;

              const ArrayList<short int> &eta_mapping =
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
void FarFieldExpansion<TKernelAux>::Init(const Vector& center,
    const TKernelAux &ka) {

  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  center_.Copy(center);
  order_ = -1;
  sea_ = &(ka.sea_);
  ka_ = &ka;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernelAux>
void FarFieldExpansion<TKernelAux>::Init(const TKernelAux &ka) {

  // copy kernel type, center, and bandwidth squared
  kernel_ = &(ka.kernel_);
  order_ = -1;
  sea_ = &(ka.sea_);
  center_.Init(sea_->get_dimension());
  center_.SetZero();
  ka_ = &ka;

  // initialize coefficient array
  coeffs_.Init(sea_->get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<typename TKernelAux>
template<typename TBound>
int FarFieldExpansion<TKernelAux>::OrderForConvolving
(const TBound &far_field_region, const Vector &far_field_region_centroid,
 const TBound &local_field_region, const Vector &local_field_region_centroid,
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

  for(index_t i = 0; i < center_.length(); i++) {
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

  for(index_t i = 0; i < total_num_coeffs; i++) {
    const ArrayList<short int> &mapping = sea_->get_multiindex(i);
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
  Vector prev_coeffs;
  Vector prev_center;
  const ArrayList <short int> *multiindex_mapping =
    sea_->get_multiindex_mapping();
  const ArrayList <short int> *lower_mapping_index =
    sea_->get_lower_mapping_index();

  ArrayList <short int> tmp_storage;
  Vector center_diff;
  Vector inv_multiindex_factorials;

  center_diff.Init(dim);

  // retrieve coefficients to be translated and helper mappings
  prev_coeffs.Alias(se.get_coeffs());
  prev_center.Alias(*(se.get_center()));
  tmp_storage.Init(sea_->get_dimension());
  inv_multiindex_factorials.Alias(sea_->get_inv_multiindex_factorials());

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

    const ArrayList <short int> &gamma_mapping = multiindex_mapping[j];
    const ArrayList <short int> &lower_mappings_for_gamma =
      lower_mapping_index[j];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(index_t k = 0; k < lower_mappings_for_gamma.size(); k++) {

      const ArrayList <short int> &inner_mapping =
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

  Vector pos_arrtmp, neg_arrtmp;
  Matrix derivative_map;
  ka_->AllocateDerivativeMap(sea_->get_dimension(), 2 * truncation_order,
                             &derivative_map);
  Vector local_center;
  Vector cent_diff;
  Vector local_coeffs;
  int local_order = se.get_order();
  int dimension = sea_->get_dimension();
  int total_num_coeffs = sea_->get_total_num_coeffs(truncation_order);
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

  // Compute derivatives.
  pos_arrtmp.Init(total_num_coeffs);
  neg_arrtmp.Init(total_num_coeffs);

  // Compute center difference divided by the bandwidth factor.
  for(index_t j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // Compute required partial derivatives.
  ka_->ComputeDirectionalDerivatives(cent_diff, &derivative_map,
                                     2 * truncation_order);
  ArrayList<short int> beta_plus_alpha;
  beta_plus_alpha.Init(dimension);

  for(index_t j = 0; j < total_num_coeffs; j++) {

    const ArrayList<short int> &beta_mapping = sea_->get_multiindex(j);
    pos_arrtmp[j] = neg_arrtmp[j] = 0;

    for(index_t k = 0; k < total_num_coeffs; k++) {

      const ArrayList<short int> &alpha_mapping = sea_->get_multiindex(k);
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

  Vector C_k_neg = sea_->get_neg_inv_multiindex_factorials();
  for(index_t j = 0; j < total_num_coeffs; j++) {
    local_coeffs[j] += (pos_arrtmp[j] + neg_arrtmp[j]) * C_k_neg[j];
  }
}

#endif
