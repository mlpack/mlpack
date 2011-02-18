/** @file multivariate_local_dev.h
 *
 *  A template instantiation of Cartesian local expansion in $O(p^D)$
 *  expansion.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_HYPERCUBE_LOCAL_DEV_H
#define MLPACK_SERIES_EXPANSION_HYPERCUBE_LOCAL_DEV_H

#include "mlpack/series_expansion/cartesian_local.h"

namespace mlpack {
namespace series_expansion {

template<>
template<typename KernelAuxType>
void CartesianLocal<mlpack::series_expansion::HYPERCUBE>::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  const core::table::DenseMatrix& data,
  const core::table::DensePoint& weights, int begin, int end, int order) {

  if(order > order_) {
    order_ = order;
  }

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);

  // get inverse factorials (precomputed)
  core::table::DensePoint neg_inv_multiindex_factorials;
  neg_inv_multiindex_factorials.Alias(
    kernel_aux_in.global().get_neg_inv_multiindex_factorials());

  // declare deritave mapping
  core::table::DenseMatrix derivative_map;
  kernel_aux_in.AllocateDerivativeMap(dim, order, &derivative_map);

  // some temporary variables
  core::table::DensePoint x_r_minus_x_Q;
  x_r_minus_x_Q.Init(dim);

  // sqrt two times bandwidth
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(order);

  // for each data point,
  for(int r = begin; r < end; r++) {

    // calculate x_r - x_Q
    for(int d = 0; d < dim; d++) {
      x_r_minus_x_Q[d] = (center_[d] - data.get(d, r)) / bandwidth_factor;
    }

    // Precompute necessary partial derivatives based on coordinate
    // difference.
    kernel_aux_in.ComputeDirectionalDerivatives(
      x_r_minus_x_Q, &derivative_map, order);

    // compute h_{beta}((x_r - x_Q) / sqrt(2h^2))
    for(int j = 0; j < total_num_coeffs; j++) {
      int index = traversal_order[j];
      const std::vector<short int> &mapping =
        kernel_aux_in.global().get_multiindex(index);
      double partial_derivative =
        kernel_aux_in.ComputePartialDerivative(derivative_map, mapping);

      // For now, the weight is uniform. Replace with the following
      // lines for non-uniform weights.
      coeffs_[index] += neg_inv_multiindex_factorials[index] *
                        partial_derivative;
      //coeffs_[index] += neg_inv_multiindex_factorials[index] * weights[r] *
      //                  partial_derivative;
    }
  } // End of looping through each reference point.
}

template<>
template<typename KernelAuxType>
void CartesianLocal <
mlpack::series_expansion::HYPERCUBE >::Print(
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
double CartesianLocal <
mlpack::series_expansion::HYPERCUBE >::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint& x_q) const {

  // if there are no local coefficients, then return 0
  if(order_ < 0) {
    return 0;
  }

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
  tmp.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  std::vector<short int> heads(dim + 1, 0);

  // compute (x_q - x_Q) / (sqrt(2h^2))
  for(int i = 0; i < dim; i++) {
    x_Q_to_x_q[i] = (x_q[i] - center_[i]) / bandwidth_factor;
  }
  heads[dim] = std::numeric_limits<short int>::max();

  tmp[0] = 1.0;

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(order_);

  for(int i = 1; i < total_num_coeffs; i++) {

    int index = traversal_order[i];
    const std::vector<short int> &lower_mappings =
      kernel_aux_in.global().lower_mapping_index(index);

    // from the direct descendant, recursively compute the multipole moments
    int direct_ancestor_mapping_pos =
      lower_mappings[lower_mappings.size() - 2];
    int position = 0;
    const std::vector<short int> &mapping =
      kernel_aux_in.global().get_multiindex(index);
    const std::vector<short int> &direct_ancestor_mapping =
      kernel_aux_in.global().get_multiindex(direct_ancestor_mapping_pos);
    for(int i = 0; i < dim; i++) {
      if(mapping[i] != direct_ancestor_mapping[i]) {
        position = i;
        break;
      }
    }
    tmp[index] = tmp[direct_ancestor_mapping_pos] * x_Q_to_x_q[position];
  }

  for(int i = 0; i < total_num_coeffs; i++) {
    int index = traversal_order[i];
    sum += coeffs_[index] * tmp[index];
  }

  return sum;
}

template<>
template<typename KernelAuxType, typename CartesianFarFieldType>
void CartesianLocal<mlpack::series_expansion::HYPERCUBE>::TranslateFromFarField(
  const KernelAuxType &kernel_aux_in,
  const CartesianFarFieldType &se) {

  core::table::DensePoint pos_arrtmp, neg_arrtmp;
  core::table::DenseMatrix derivative_map;
  core::table::DensePoint far_center;
  core::table::DensePoint cent_diff;
  core::table::DensePoint far_coeffs;
  int dimension = kernel_aux_in.global().get_dimension();
  int far_order = se.get_order();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(far_order);
  int limit;
  double bandwidth_factor = kernel_aux_in.BandwidthFactor(se.bandwidth_sq());

  kernel_aux_in.AllocateDerivativeMap(dimension, 2 * order_, &derivative_map);

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

  // get the order of traversal for the given order of approximation
  std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping_[far_order];

  for(int j = 0; j < total_num_coeffs; j++) {

    int index_j = traversal_order[j];
    const std::vector<short int> &beta_mapping =
      kernel_aux_in.global().get_multiindex(index_j);
    pos_arrtmp[index_j] = neg_arrtmp[index_j] = 0;

    for(int k = 0; k < total_num_coeffs; k++) {

      int index_k = traversal_order[k];
      const std::vector<short int> &alpha_mapping =
        kernel_aux_in.global().get_multiindex(index_k);
      for(int d = 0; d < dimension; d++) {
        beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
        kernel_aux_in.ComputePartialDerivative(derivative_map, beta_plus_alpha);

      double prod = far_coeffs[index_k] * derivative_factor;

      if(prod > 0) {
        pos_arrtmp[index_j] += prod;
      }
      else {
        neg_arrtmp[index_j] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  const core::table::DensePoint &C_k_neg =
    kernel_aux_in.global().get_neg_inv_multiindex_factorials();
  for(int j = 0; j < total_num_coeffs; j++) {
    int index_j = traversal_order[j];
    coeffs_[index_j] += (pos_arrtmp[index_j] + neg_arrtmp[index_j]) *
                        C_k_neg[index_j];
  }
}

template<>
template<typename KernelAuxType>
void CartesianLocal <
mlpack::series_expansion::HYPERCUBE >::TranslateToLocal(
  const KernelAuxType &kernel_aux_in,
  CartesianLocal<mlpack::series_expansion::HYPERCUBE> *se) const {

  // if no local coefficients have formed, then nothing to translate
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

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(order_);

  // do the actual translation
  for(int j = 0; j < total_num_coeffs; j++) {

    int index_j = traversal_order[j];
    const std::vector<short int> &alpha_mapping =
      kernel_aux_in.global().get_multiindex(index_j);
    const std::vector<short int> &upper_mappings_for_alpha =
      upper_mapping_index[index_j];
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

      if(flag) {
        continue;
      }

      for(int l = 0; l < dim; l++) {
        diff1 *= pow(center_diff[l], tmp_storage[l]);
      }

      double prod = coeffs_[upper_mappings_for_alpha[k]] * diff1 *
                    kernel_aux_in.global().get_n_multichoose_k_by_pos
                    (upper_mappings_for_alpha[k], index_j);

      if(prod > 0) {
        pos_coeffs += prod;
      }
      else {
        neg_coeffs += prod;
      }

    } // end of k loop

    new_coeffs[index_j] += pos_coeffs + neg_coeffs;
  } // end of j loop
}
}
}

#endif
