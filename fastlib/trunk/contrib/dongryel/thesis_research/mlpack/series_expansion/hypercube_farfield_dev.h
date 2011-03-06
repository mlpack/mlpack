/** @file hypercube_farfield_dev.h
 *
 *  This file contains an implementation of $O(p^D)$ expansion for
 *  computing the coefficients for a far-field expansion for an
 *  arbitrary kernel function. This is a template specialization of
 *  CartesianFarField class.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_HYPERCUBE_FARFIELD_DEV_H
#define MLPACK_SERIES_EXPANSION_HYPERCUBE_FARFIELD_DEV_H

#include <vector>
#include "mlpack/series_expansion/cartesian_farfield.h"

namespace mlpack {
namespace series_expansion {

template<>
template<typename KernelAuxType, typename TreeIteratorType>
void CartesianFarField<mlpack::series_expansion::HYPERCUBE>::AccumulateCoeffs(
  const KernelAuxType &kernel_aux_in,
  TreeIteratorType &it, int order) {

  int dim = kernel_aux_in.global().get_dimension();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  int max_total_num_coeffs = kernel_aux_in.global().get_max_total_num_coeffs();
  core::table::DensePoint x_r, tmp;
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());

  // initialize temporary variables
  x_r.Init(dim);
  tmp.Init(max_total_num_coeffs);
  core::table::DensePoint pos_coeffs;
  core::table::DensePoint neg_coeffs;
  pos_coeffs.Init(max_total_num_coeffs);
  pos_coeffs.SetZero();
  neg_coeffs.Init(max_total_num_coeffs);
  neg_coeffs.SetZero();

  // set to new order if greater
  if(order_ < order) {
    order_ = order;
  }

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(order_);

  // Repeat for each reference point in this reference node.
  while(it.HasNext()) {

    // Calculate the coordinate difference between the ref point and
    // the centroid.
    core::table::DensePoint point;
    double weight;
    it.Next(&point, &weight);
    for(int i = 0; i < dim; i++) {
      x_r[i] = (point[i] - center_[i]) / bandwidth_factor;
    }

    tmp.SetZero();
    tmp[0] = 1.0;

    for(int i = 1; i < total_num_coeffs; i++) {

      int index = traversal_order[i];
      const std::vector<short int> &lower_mappings =
        kernel_aux_in.global().get_lower_mapping_index()[index];

      // from the direct descendant, recursively compute the multipole moments
      int direct_ancestor_mapping_pos =
        lower_mappings[lower_mappings.size() - 2];

      int position = 0;
      const std::vector<short int> &mapping =
        kernel_aux_in.global().get_multiindex_mapping()[index];
      const std::vector<short int> &direct_ancestor_mapping =
        kernel_aux_in.global().get_multiindex_mapping()[
          direct_ancestor_mapping_pos];
      for(int i = 0; i < dim; i++) {
        if(mapping[i] != direct_ancestor_mapping[i]) {
          position = i;
          break;
        }
      }
      tmp[index] = tmp[direct_ancestor_mapping_pos] * x_r[position];
    }

    // Tally up the result in A_k.
    for(int i = 0; i < total_num_coeffs; i++) {

      int index = traversal_order[i];
      double prod = weight * tmp[index];

      if(prod > 0) {
        pos_coeffs[index] += prod;
      }
      else {
        neg_coeffs[index] += prod;
      }
    }

  } // End of looping through each reference point

  for(int r = 0; r < total_num_coeffs; r++) {
    int index = traversal_order[r];
    coeffs_[index] += (pos_coeffs[index] + neg_coeffs[index]) *
                      kernel_aux_in.global().
                      get_inv_multiindex_factorials()[index];
  }
}

template<>
template<typename KernelAuxType>
double CartesianFarField<mlpack::series_expansion::HYPERCUBE>::EvaluateField(
  const KernelAuxType &kernel_aux_in,
  const core::table::DensePoint &x_q, int order) const {

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
  kernel_aux_in.AllocateDerivativeMap(dim, order_, &derivative_map);

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
    x_q_minus_x_R, &derivative_map, order_);

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(order_);

  // compute h_{\alpha}((x_q - x_R)/sqrt(2h^2)) ((x_r - x_R)/h)^{\alpha}
  for(int j = 0; j < total_num_coeffs; j++) {

    int index = traversal_order[j];
    const std::vector<short int> &mapping =
      kernel_aux_in.global().get_multiindex(index);
    double arrtmp =
      kernel_aux_in.ComputePartialDerivative(derivative_map, mapping);
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

template<>
template<typename KernelAuxType>
void CartesianFarField<mlpack::series_expansion::HYPERCUBE>::Print(
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

template<>
template<typename KernelAuxType>
void CartesianFarField <
mlpack::series_expansion::HYPERCUBE >::TranslateFromFarField(
  const KernelAuxType &kernel_aux_in, const CartesianFarField &se) {

  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(kernel_aux_in.kernel().bandwidth_sq());
  int dim = kernel_aux_in.global().get_dimension();
  int order = se.get_order();
  int total_num_coeffs = kernel_aux_in.global().get_total_num_coeffs(order);
  core::table::DensePoint prev_coeffs;
  core::table::DensePoint prev_center;
  const std::vector< std::vector<short int> > &multiindex_mapping =
    kernel_aux_in.global().get_multiindex_mapping();
  const std::vector< std::vector<short int> > &lower_mapping_index =
    kernel_aux_in.global().get_lower_mapping_index();

  std::vector<short int> tmp_storage;
  core::table::DensePoint center_diff;
  core::table::DensePoint inv_multiindex_factorials;

  center_diff.Init(dim);

  // retrieve coefficients to be translated and helper mappings
  prev_coeffs.Alias(se.get_coeffs());
  prev_center.Alias(se.get_center());
  tmp_storage.resize(kernel_aux_in.global().get_dimension());
  inv_multiindex_factorials.Alias(
    kernel_aux_in.global().get_inv_multiindex_factorials());

  // no coefficients can be translated
  if(order == -1) {
    return;
  }
  else {
    order_ = order;
  }

  // compute center difference
  for(int j = 0; j < dim; j++) {
    center_diff[j] = prev_center[j] - center_[j];
  }

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(order);

  for(int j = 0; j < total_num_coeffs; j++) {

    int index = traversal_order[j];
    const std::vector<short int> &gamma_mapping = multiindex_mapping[index];
    const std::vector<short int> &lower_mappings_for_gamma =
      lower_mapping_index[index];
    double pos_coeff = 0;
    double neg_coeff = 0;

    for(unsigned int k = 0; k < lower_mappings_for_gamma.size(); k++) {

      const std::vector<short int> &inner_mapping =
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

      double prod = prev_coeffs[lower_mappings_for_gamma[k]] * diff1 *
                    inv_multiindex_factorials
                    [kernel_aux_in.global().ComputeMultiindexPosition(tmp_storage)];

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

template<>
template<typename KernelAuxType, typename CartesianLocalType>
void CartesianFarField<mlpack::series_expansion::HYPERCUBE>::TranslateToLocal(
  const KernelAuxType &kernel_aux_in,
  int truncation_order, CartesianLocalType *se) const {

  core::table::DensePoint pos_arrtmp, neg_arrtmp;
  core::table::DenseMatrix derivative_map;
  core::table::DensePoint local_center;
  core::table::DensePoint cent_diff;
  core::table::DensePoint local_coeffs;
  int local_order = se->get_order();
  int dimension = kernel_aux_in.global().get_dimension();
  int total_num_coeffs =
    kernel_aux_in.global().get_total_num_coeffs(truncation_order);
  double bandwidth_factor =
    kernel_aux_in.BandwidthFactor(
      kernel_aux_in.kernel().bandwidth_sq());

  kernel_aux_in.AllocateDerivativeMap(
    dimension, 2 * truncation_order, &derivative_map);

  // get center and coefficients for local expansion
  local_center.Alias(se->get_center());
  local_coeffs.Alias(se->get_coeffs());
  cent_diff.Init(dimension);

  // if the order of the far field expansion is greater than the
  // local one we are adding onto, then increase the order.
  if(local_order < truncation_order) {
    se->set_order(truncation_order);
  }

  // compute Gaussian derivative
  pos_arrtmp.Init(kernel_aux_in.global().get_max_total_num_coeffs());
  neg_arrtmp.Init(kernel_aux_in.global().get_max_total_num_coeffs());

  // compute center difference divided by bw_times_sqrt_two;
  for(int j = 0; j < dimension; j++) {
    cent_diff[j] = (local_center[j] - center_[j]) / bandwidth_factor;
  }

  // compute required partial derivatives
  kernel_aux_in.ComputeDirectionalDerivatives(cent_diff, &derivative_map,
      2 * truncation_order);
  std::vector<short int> beta_plus_alpha(dimension);

  // get the order of traversal for the given order of approximation
  const std::vector<short int> &traversal_order =
    kernel_aux_in.global().traversal_mapping(truncation_order);

  for(int j = 0; j < total_num_coeffs; j++) {

    int index = traversal_order[j];
    const std::vector<short int> &beta_mapping =
      kernel_aux_in.global().get_multiindex(index);
    pos_arrtmp[index] = neg_arrtmp[index] = 0;

    for(int k = 0; k < total_num_coeffs; k++) {

      int index_k = traversal_order[k];

      const std::vector<short int> &alpha_mapping =
        kernel_aux_in.global().get_multiindex(index_k);
      for(int d = 0; d < dimension; d++) {
        beta_plus_alpha[d] = beta_mapping[d] + alpha_mapping[d];
      }
      double derivative_factor =
        kernel_aux_in.ComputePartialDerivative(derivative_map, beta_plus_alpha);

      double prod = coeffs_[index_k] * derivative_factor;

      if(prod > 0) {
        pos_arrtmp[index] += prod;
      }
      else {
        neg_arrtmp[index] += prod;
      }
    } // end of k-loop
  } // end of j-loop

  const core::table::DensePoint &C_k_neg =
    kernel_aux_in.global().get_neg_inv_multiindex_factorials();
  for(int j = 0; j < total_num_coeffs; j++) {
    int index = traversal_order[j];
    local_coeffs[index] += (pos_arrtmp[index] + neg_arrtmp[index]) *
                           C_k_neg[index];
  }
}
}
}

#endif
