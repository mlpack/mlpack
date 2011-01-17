/** @file cartesian_farfield_dev.h
 *
 *  This file contains an implementation of $O(D^p)$ expansion for
 *  computing the coefficients for a far-field expansion for an
 *  arbitrary kernel function
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_CARTESIAN_FARFIELD_DEV_H
#define MLPACK_SERIES_EXPANSION_CARTESIAN_FARFIELD_DEV_H

#include "mlpack/series_expansion/cartesian_expansion_global.h"
#include "mlpack/series_expansion/cartesian_farfield.h"

namespace mlpack {
namespace series_expansion {

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
core::table::DensePoint &CartesianFarField<ExpansionType>::get_center() {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint &CartesianFarField <
ExpansionType >::get_center() const {
  return center_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint &CartesianFarField <
ExpansionType >::get_coeffs() const {
  return coeffs_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
short int CartesianFarField<ExpansionType>::get_order() const {
  return order_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianFarField<ExpansionType>::get_weight_sum() const {
  return coeffs_[0];
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianFarField<ExpansionType>::set_order(short int new_order) {
  order_ = new_order;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianFarField<ExpansionType>::set_center(
  const core::table::DensePoint &center) {
  for(int i = 0; i < center.length(); i++) {
    center_[i] = center[i];
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::Accumulate(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &v, double weight, int order) {

  int dim = v.length();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  core::table::DensePoint tmp;
  int r, i, j, k, t, tail;
  std::vector<short int> heads(dim + 1, 0);
  core::table::DensePoint x_r;
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  x_r.Init(dim);
  core::table::DensePoint pos_coeffs;
  core::table::DensePoint neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  core::table::DensePoint C_k;

  // Calculate the coordinate difference between the ref point and the
  // centroid.
  for(i = 0; i < dim; i++) {
    x_r[i] = (v[i] - center_[i]) / bandwidth_factor;
  }

  // initialize heads
  heads[dim] = std::numeric_limits<short int>::max();

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
  C_k.Alias(kernel_aux_in.global().get_inv_multiindex_factorials());

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix& data,
  const core::table::DensePoint& weights,
  int begin, int end, int order) {

  int dim = data.n_rows();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  core::table::DensePoint tmp;
  int r, i, j, k, t, tail;
  std::vector<short int> heads(dim + 1, 0);
  core::table::DensePoint x_r;
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // initialize temporary variables
  tmp.Init(total_num_coeffs);
  x_r.Init(dim);
  core::table::DensePoint pos_coeffs;
  core::table::DensePoint neg_coeffs;
  pos_coeffs.Init(total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }
  core::table::DensePoint C_k;

  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {

    // Calculate the coordinate difference between the ref point and the
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
    }

    // initialize heads
    heads[dim] = std::numeric_limits<short int>::max();

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
  C_k.Alias(kernel_aux_in.global().get_inv_multiindex_factorials());

  for(r = 0; r < total_num_coeffs; r++) {
    coeffs_[r] += (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::RefineCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix &data,
  const core::table::DensePoint &weights,
  int begin, int end, int order) {

  if(order_ < 0) {
    AccumulateCoeffs(data, weights, begin, end, order);
    return;
  }

  int dim = data.n_rows();
  int old_total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order_);
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  double tmp;
  int r, i, j;
  core::table::DensePoint x_r;
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // initialize temporary variables
  x_r.Init(dim);
  core::table::DensePoint pos_coeffs;
  core::table::DensePoint neg_coeffs;
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

  core::table::DensePoint C_k;
  // Repeat for each reference point in this reference node.
  for(r = begin; r < end; r++) {

    // Calculate the coordinate difference between the ref point and the
    // centroid.
    for(i = 0; i < dim; i++) {
      x_r[i] = (data.get(i, r) - center_[i]) / bandwidth_factor;
    }

    // compute in bruteforce way
    for(i = old_total_num_coeffs; i < total_num_coeffs; i++) {
      const std::vector<short int> &mapping =
        kernel_aux_in.global().get_multiindex(i);
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
  C_k.Alias(kernel_aux_in.global().get_inv_multiindex_factorials());

  for(r = old_total_num_coeffs; r < total_num_coeffs; r++) {
    coeffs_[r] = (pos_coeffs[r] + neg_coeffs[r]) * C_k[r];
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
double CartesianFarField<ExpansionType>::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix& data, int row_num, int order) const {
  return EvaluateField(kernel_aux_in, data.GetColumnPtr(row_num), order);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
double CartesianFarField<ExpansionType>::EvaluateField(
  const KernelAuxType &kernel_aux_in, const double *x_q, int order) const {

  // dimension
  int dim = kernel_aux_in.global().get_dimension();

  // total number of coefficients
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);

  // square root times bandwidth
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // the evaluated sum
  double pos_multipole_sum = 0;
  double neg_multipole_sum = 0;
  double multipole_sum = 0;

  // computed derivative map
  core::table::DenseMatrix derivative_map;
  kernel_aux_in.AllocateDerivativeMap(dim, order, &derivative_map);

  // temporary variable
  core::table::DensePoint arrtmp;
  arrtmp.Init(total_num_coeffs);

  // (x_q - x_R) scaled by bandwidth
  core::table::DensePoint x_q_minus_x_R;
  x_q_minus_x_R.Init(dim);

  // compute (x_q - x_R) / (sqrt(2h^2))
  for(int d = 0; d < dim; d++) {
    x_q_minus_x_R[d] = (x_q[d] - center_[d]) / bandwidth_factor;
  }

  // compute deriative maps based on coordinate difference.
  kernel_aux_in.ComputeDirectionalDerivatives(
    x_q_minus_x_R, &derivative_map, order);

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(int j = 0; j < total_num_coeffs; j++) {
    const std::vector<short int> &mapping = kernel_aux_in.global().get_multiindex(j);
    double arrtmp =
      kernel_aux_in.ComputePartialDerivative(derivative_map, mapping);
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

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::Init(
  const KernelAuxType &kernel_aux_in, const core::table::DensePoint& center) {

  // Copy the center.
  center_.Copy(center);
  order_ = -1;

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename TKernelAux>
void CartesianFarField<ExpansionType>::Init(const TKernelAux &kernel_aux_in) {

  // Set the center to be a zero vector.
  order_ = -1;
  center_.Init(kernel_aux_in.global().get_dimension());
  center_.SetZero();

  // Initialize coefficient array.
  coeffs_.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  coeffs_.SetZero();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType, typename BoundType>
int CartesianFarField<ExpansionType>::OrderForEvaluating(
  const KernelAuxType &kernel_aux_in,
  const BoundType &far_field_region,
  const BoundType &local_field_region, double min_dist_sqd_regions,
  double max_dist_sqd_regions, double max_error, double *actual_error) const {

  return kernel_aux_in.OrderForEvaluatingFarField(
           far_field_region,
           local_field_region,
           min_dist_sqd_regions,
           max_dist_sqd_regions, max_error,
           actual_error);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType, typename BoundType>
int CartesianFarField<ExpansionType>::OrderForConvertingToLocal(
  const KernelAuxType &kernel_aux_in,
  const BoundType &far_field_region, const BoundType &local_field_region,
  double min_dist_sqd_regions, double max_dist_sqd_regions, double max_error,
  double *actual_error) const {

  return kernel_aux_in.OrderForConvertingFromFarFieldToLocal(
           far_field_region,
           local_field_region,
           min_dist_sqd_regions,
           max_dist_sqd_regions,
           max_error, actual_error);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::Print(
  const KernelAuxType &kernel_aux_in, const char *name, FILE *stream) const {

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Far field expansion\n");
  fprintf(stream, "Center: ");

  for(int i = 0; i < center_.length(); i++) {
    fprintf(stream, "%g ", center_[i]);
  }
  fprintf(stream, "\n");

  fprintf(stream, "f(");
  for(int d = 0; d < dim; d++) {
    fprintf(stream, "x_q%d", d);
    if(d < dim - 1)
      fprintf(stream, ",");
  }
  fprintf(stream, ") = \\sum\\limits_{x_r \\in R} K(||x_q - x_r||) = ");

  for(int i = 0; i < total_num_coeffs; i++) {
    const std::vector<short int> &mapping =
      kernel_aux_in.global().get_multiindex(i);
    fprintf(stream, "%g ", coeffs_[i]);

    fprintf(stream, "(-1)^(");
    for(int d = 0; d < dim; d++) {
      fprintf(stream, "%d", mapping[d]);
      if(d < dim - 1)
        fprintf(stream, " + ");
    }
    fprintf(stream, ") D^((");
    for(int d = 0; d < dim; d++) {
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

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::TranslateFromFarField(
  const KernelAuxType &kernel_aux_in, const CartesianFarField &se) {

  double bandwidth_factor = kernel_aux_in.BandwidthFactor(se.bandwidth_sq());
  int dim = kernel_aux_in.global().get_dimension();
  int order = se.get_order();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  core::table::DensePoint prev_coeffs;
  core::table::DensePoint prev_center;
  const std::vector <short int> *multiindex_mapping =
    kernel_aux_in.global().get_multiindex_mapping();
  const std::vector <short int> *lower_mapping_index =
    kernel_aux_in.global().get_lower_mapping_index();

  std::vector <short int> tmp_storage;
  core::table::DensePoint center_diff;
  core::table::DensePoint inv_multiindex_factorials;

  center_diff.Init(dim);

  // retrieve coefficients to be translated and helper mappings
  prev_coeffs.Alias(se.get_coeffs());
  prev_center.Alias(*(se.get_center()));
  tmp_storage.resize(kernel_aux_in.global().get_dimension());
  inv_multiindex_factorials.Alias(
    kernel_aux_in.global().get_inv_multiindex_factorials());

  // no coefficients can be translated
  if(order == -1)
    return;
  else
    order_ = order;

  // compute center difference
  for(int j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  for(int j = 0; j < total_num_coeffs; j++) {

    const std::vector <short int> &gamma_mapping = multiindex_mapping[j];
    const std::vector <short int> &lower_mappings_for_gamma =
      lower_mapping_index[j];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(int k = 0; k < lower_mappings_for_gamma.size(); k++) {

      const std::vector <short int> &inner_mapping =
        multiindex_mapping[lower_mappings_for_gamma[k]];

      int flag = 0;
      double diff1;

      // compute gamma minus alpha
      for(int l = 0; l < dim; l++) {
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

      for(int l = 0; l < dim; l++) {
        diff1 *= pow(center_diff[l] / bandwidth_factor, tmp_storage[l]);
      }

      double prod =
        prev_coeffs[lower_mappings_for_gamma[k]] * diff1 *
        inv_multiindex_factorials[
          kernel_aux_in.global().ComputeMultiindexPosition(tmp_storage)];

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

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
template<typename KernelAuxType>
void CartesianFarField<ExpansionType>::TranslateToLocal(
  const KernelAuxType &kernel_aux_in, int truncation_order,
  CartesianLocal<ExpansionType> &se) {

  core::table::DensePoint pos_arrtmp, neg_arrtmp;
  core::table::DenseMatrix derivative_map;
  kernel_aux_in.AllocateDerivativeMap(
    kernel_aux_in.global().get_dimension(), 2 * truncation_order,
    &derivative_map);
  core::table::DensePoint local_center;
  core::table::DensePoint cent_diff;
  core::table::DensePoint local_coeffs;
  int local_order = se.get_order();
  int dimension = kernel_aux_in.global().get_dimension();
  int total_num_coeffs =
    kernel_aux_in.global().get_total_num_coeffs(truncation_order);
  double bandwidth_factor = kernel_aux_in.BandwidthFactor(se.bandwidth_sq());

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
  for(int j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // Compute required partial derivatives.
  kernel_aux_in.ComputeDirectionalDerivatives(cent_diff, &derivative_map,
      2 * truncation_order);
  std::vector<short int> beta_plus_alpha;
  beta_plus_alpha.Init(dimension);

  for(int j = 0; j < total_num_coeffs; j++) {

    const std::vector<short int> &beta_mapping =
      kernel_aux_in.global().get_multiindex(j);
    pos_arrtmp[j] = neg_arrtmp[j] = 0;

    for(int k = 0; k < total_num_coeffs; k++) {

      const std::vector<short int> &alpha_mapping =
        kernel_aux_in.global().get_multiindex(k);
      for(int d = 0; d < dimension; d++) {
        beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
        kernel_aux_in.ComputePartialDerivative(derivative_map, beta_plus_alpha);

      double prod = coeffs_[k] * derivative_factor;

      if(prod > 0) {
        pos_arrtmp[j] += prod;
      }
      else {
        neg_arrtmp[j] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  core::table::DensePoint C_k_neg =
    kernel_aux_in.global().get_neg_inv_multiindex_factorials();
  for(int j = 0; j < total_num_coeffs; j++) {
    local_coeffs[j] += (pos_arrtmp[j] + neg_arrtmp[j]) * C_k_neg[j];
  }
}
}
}

#endif
