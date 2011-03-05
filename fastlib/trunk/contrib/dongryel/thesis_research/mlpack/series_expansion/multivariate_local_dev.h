/** @file multivariate_local_dev.h
 *
 *  A template instantiation of Cartesian local expansion in $O(D^p)$
 *  expansion.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_MULTIVARIATE_LOCAL_DEV_H
#define MLPACK_SERIES_EXPANSION_MULTIVARIATE_LOCAL_DEV_H

#include "mlpack/series_expansion/cartesian_local.h"

namespace mlpack {
namespace series_expansion {

template<>
template<typename KernelAuxType, typename TreeIteratorType>
void CartesianLocal <
mlpack::series_expansion::MULTIVARIATE >::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix& weights,
  TreeIteratorType &it, int order) {

  if(order > order_) {
    order_ = order;
  }

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);

  // Get inverse factorials (precomputed).
  core::table::DensePoint neg_inv_multiindex_factorials;
  neg_inv_multiindex_factorials.Alias
  (kernel_aux_in.global().get_neg_inv_multiindex_factorials());

  // Declare deritave mapping.
  core::table::DenseMatrix derivative_map;
  kernel_aux_in.AllocateDerivativeMap(dim, order, &derivative_map);

  // Some temporary variables.
  core::table::DensePoint arrtmp;
  arrtmp.Init(total_num_coeffs);
  core::table::DensePoint x_r_minus_x_Q;
  x_r_minus_x_Q.Init(dim);

  // The bandwidth factor to be divided along each dimension.
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // for each data point,
  while(it.HasNext()) {

    // Get the reference point.
    core::table::DensePoint point;
    it.Next(&point);

    // calculate x_r - x_Q
    for(int d = 0; d < dim; d++) {
      x_r_minus_x_Q[d] = (center_[d] - point[d]) / bandwidth_factor;
    }

    // precompute necessary partial derivatives based on coordinate difference
    kernel_aux_in.ComputeDirectionalDerivatives(
      x_r_minus_x_Q, &derivative_map, order);

    // compute h_{beta}((x_r - x_Q) / sqrt(2h^2))
    for(int j = 0; j < total_num_coeffs; j++) {
      const std::vector<short int> &mapping =
        kernel_aux_in.global().get_multiindex(j);
      arrtmp[j] = kernel_aux_in.ComputePartialDerivative(
                    derivative_map, mapping);
    }

    for(int j = 0; j < total_num_coeffs; j++) {

      // Replace it with the following line for non-uniform case.
      coeffs_[j] += neg_inv_multiindex_factorials[j] * arrtmp[j];
      //coeffs_[j] += neg_inv_multiindex_factorials[j] * weights[r] *
      //              arrtmp[j];
    }
  } // End of looping through each reference point.
}

template<>
template<typename KernelAuxType>
void CartesianLocal<mlpack::series_expansion::MULTIVARIATE>::Print(
  const KernelAuxType &kernel_aux_in, const char *name, FILE *stream) const {

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order_);

  fprintf(stream, "----- SERIESEXPANSION %s ------\n", name);
  fprintf(stream, "Local expansion\n");
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
    fprintf(stream, "%g", coeffs_[i]);

    for(int d = 0; d < dim; d++) {
      fprintf(stream, "(x_q%d - (%g))^%d ", d, center_[d], mapping[d]);
    }

    if(i < total_num_coeffs - 1) {
      fprintf(stream, " + ");
    }
  }
  fprintf(stream, "\n");
}

template<>
template<typename KernelAuxType>
double CartesianLocal<mlpack::series_expansion::MULTIVARIATE>::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &x_q) const {

  // if there are no local expansion here, then return 0
  if(order_ < 0) {
    return 0;
  }

  int k, t, tail;

  // total number of coefficient
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order_);

  // number of dimensions
  int dim = kernel_aux_in.global().get_dimension();

  // evaluated sum to be returned
  double sum = 0;

  // sqrt two bandwidth
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // temporary variable
  core::table::DensePoint x_Q_to_x_q;
  x_Q_to_x_q.Init(dim);
  core::table::DensePoint tmp;
  tmp.Init(total_num_coeffs);
  std::vector<short int> heads(dim + 1, 0);

  // compute (x_q - x_Q) / (sqrt(2h^2))
  for(int i = 0; i < dim; i++) {
    x_Q_to_x_q[i] = (x_q[i] - center_[i]) / bandwidth_factor;
  }
  heads[dim] = std::numeric_limits<short int>::max();

  tmp[0] = 1.0;

  for(k = 1, t = 1, tail = 1; k <= order_; k++, tail = t) {

    for(int i = 0; i < dim; i++) {
      int head = heads[i];
      heads[i] = t;

      for(int j = head; j < tail; j++, t++) {
        tmp[t] = tmp[j] * x_Q_to_x_q[i];
      }
    }
  }

  for(int i = 0; i < total_num_coeffs; i++) {
    sum += coeffs_[i] * tmp[i];
  }
  return sum;
}

template<>
template<typename KernelAuxType, typename CartesianFarFieldType>
void CartesianLocal <
mlpack::series_expansion::MULTIVARIATE >::TranslateFromFarField(
  const KernelAuxType &kernel_aux_in,
  const CartesianFarFieldType &se) {

  core::table::DensePoint pos_arrtmp, neg_arrtmp;
  core::table::DenseMatrix derivative_map;
  core::table::DensePoint far_center;
  core::table::DensePoint cent_diff;
  core::table::DensePoint far_coeffs;
  int dimension = kernel_aux_in.global().get_dimension();
  kernel_aux_in.AllocateDerivativeMap(dimension, 2 * order_, &derivative_map);

  int far_order = se.get_order();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(far_order);
  int limit;
  double bandwidth_factor = kernel_aux_in.BandwidthFactor(se.bandwidth_sq());

  // get center and coefficients for far field expansion
  far_center.Alias(*(se.get_center()));
  far_coeffs.Alias(se.get_coeffs());
  cent_diff.Init(dimension);

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(far_order > order_) {
    order_ = far_order;
  }

  // compute Gaussian derivative
  pos_arrtmp.Init(total_num_coeffs);
  neg_arrtmp.Init(total_num_coeffs);

  // compute center difference divided by bw_times_sqrt_two;
  for(int j = 0; j < dimension; j++) {
    cent_diff[j] = (center_[j] - far_center[j]) / bandwidth_factor;
  }

  // compute required partial derivatives
  kernel_aux_in.ComputeDirectionalDerivatives(
    cent_diff, &derivative_map, 2 * order_);
  std::vector<short int> beta_plus_alpha(dimension);

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

      double prod = far_coeffs[k] * derivative_factor;

      if(prod > 0) {
        pos_arrtmp[j] += prod;
      }
      else {
        neg_arrtmp[j] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  const core::table::DensePoint &C_k_neg =
    kernel_aux_in.global().get_neg_inv_multiindex_factorials();
  for(int j = 0; j < total_num_coeffs; j++) {
    coeffs_[j] += (pos_arrtmp[j] + neg_arrtmp[j]) * C_k_neg[j];
  }
}

template<>
template<typename KernelAuxType>
void CartesianLocal <
mlpack::series_expansion::MULTIVARIATE >::TranslateToLocal(
  const KernelAuxType &kernel_aux_in,
  CartesianLocal<mlpack::series_expansion::MULTIVARIATE> *se) const {

  // if there are no local coefficients to translate, return
  if(order_ < 0) {
    return;
  }

  // get the center and the order and the total number of coefficients of
  // the expansion we are translating from. Also get coefficients we
  // are translating
  core::table::DensePoint new_center;
  new_center.Alias(se->get_center());
  int prev_order = se->get_order();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order_);
  const std::vector< std::vector<short int> > &upper_mapping_index =
    kernel_aux_in.global().get_upper_mapping_index();
  core::table::DensePoint new_coeffs;
  new_coeffs.Alias(se->get_coeffs());

  // dimension
  int dim = kernel_aux_in.global().get_dimension();

  // temporary variable
  std::vector<short int> tmp_storage(dim);

  // sqrt two times bandwidth
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // center difference between the old center and the new one
  core::table::DensePoint center_diff;
  center_diff.Init(dim);
  for(int d = 0; d < dim; d++) {
    center_diff[d] = (new_center[d] - center_[d]) / bandwidth_factor;
  }

  // set to the new order if the order of the expansion we are translating
  // from is higher
  if(prev_order < order_) {
    se->set_order(order_);
  }

  // inverse multiindex factorials
  core::table::DensePoint C_k;
  C_k.Alias(kernel_aux_in.global().get_inv_multiindex_factorials());

  // do the actual translation
  for(int j = 0; j < total_num_coeffs; j++) {

    const std::vector<short int> &alpha_mapping =
      kernel_aux_in.global().get_multiindex(j);
    const std::vector<short int> &upper_mappings_for_alpha =
      upper_mapping_index[j];
    double pos_coeffs = 0;
    double neg_coeffs = 0;

    for(unsigned int k = 0; k < upper_mappings_for_alpha.size(); k++) {

      if(upper_mappings_for_alpha[k] >= total_num_coeffs) {
        break;
      }

      const std::vector<short int> &beta_mapping =
        kernel_aux_in.global().get_multiindex(upper_mappings_for_alpha[k]);
      int flag = 0;
      double diff1 = 1.0;

      for(int l = 0; l < dim; l++) {
        tmp_storage[l] = beta_mapping[l] - alpha_mapping[l];

        if(tmp_storage[l] < 0) {
          flag = 1;
          break;
        }
      } // end of looping over dimension

      if(flag)
        continue;

      for(int l = 0; l < dim; l++) {
        diff1 *= pow(center_diff[l], tmp_storage[l]);
      }

      double prod =
        coeffs_[upper_mappings_for_alpha[k]] * diff1 *
        kernel_aux_in.global().get_n_multichoose_k_by_pos(
          upper_mappings_for_alpha[k], j);

      if(prod > 0) {
        pos_coeffs += prod;
      }
      else {
        neg_coeffs += prod;
      }

    } // end of k loop

    new_coeffs[j] += pos_coeffs + neg_coeffs;
  } // end of j loop
}
}
}

#endif

