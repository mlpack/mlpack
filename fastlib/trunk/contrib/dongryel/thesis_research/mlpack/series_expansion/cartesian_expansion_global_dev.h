/** @file cartesian_expansion_global.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_SERIES_EXPANSION_SERIES_EXPANSION_GLOBAL_DEV_H
#define MLPACK_SERIES_EXPANSION_SERIES_EXPANSION_GLOBAL_DEV_H

#include <assert.h>
#include "mlpack/series_expansion/cartesian_expansion_global.h"

namespace mlpack {
namespace series_expansion {

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianExpansionGlobal <ExpansionType >::get_dimension() const {
  return dim_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianExpansionGlobal <ExpansionType >::factorial(int k) const {
  return factorials_[k];
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianExpansionGlobal <ExpansionType >::
ComputeUpperMappingIndex() {

  int limit = 2 * max_order_;
  std::vector<int> diff;
  diff.resize(dim_);

  // initialize the index
  upper_mapping_index_.resize(list_total_num_coeffs_[limit]);

  for(int i = 0; i < list_total_num_coeffs_[limit]; i++) {
    const std::vector<short int> &outer_mapping = multiindex_mapping_[i];

    for(int j = 0; j < list_total_num_coeffs_[limit]; j++) {
      const std::vector<short int> &inner_mapping = multiindex_mapping_[j];
      int flag = 0;

      for(int d = 0; d < dim_; d++) {
        diff[d] = inner_mapping[d] - outer_mapping[d];

        if(diff[d] < 0) {
          flag = 1;
          break;
        }
      }

      if(flag == 0) {
        (upper_mapping_index_[i]).push_back(j);
      }
    } // end of j-loop
  } // end of i-loop
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianExpansionGlobal <ExpansionType>::ComputeFactorials() {
  factorials_.Init(2 * max_order_ + 1);

  factorials_[0] = 1;
  for(int t = 1; t < factorials_.length(); t++) {
    factorials_[t] = t * factorials_[t - 1];
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianExpansionGlobal <ExpansionType>::ComputeLowerMappingIndex() {

  std::vector<int> diff(dim_);
  int limit = 2 * max_order_;

  // initialize the index
  lower_mapping_index_.resize(list_total_num_coeffs_[limit]);

  for(int i = 0; i < list_total_num_coeffs_[limit]; i++) {
    const std::vector<short int> &outer_mapping = multiindex_mapping_[i];
    for(int j = 0; j < list_total_num_coeffs_[limit]; j++) {
      const std::vector<short int> &inner_mapping = multiindex_mapping_[j];
      int flag = 0;

      for(int d = 0; d < dim_; d++) {
        diff[d] = outer_mapping[d] - inner_mapping[d];

        if(diff[d] < 0) {
          flag = 1;
          break;
        }
      }

      if(flag == 0) {
        (lower_mapping_index_[i]).push_back(j);
      }
    } // end of j-loop
  } // end of i-loop
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianExpansionGlobal <ExpansionType>::ComputeMultiindexCombination() {

  int limit = 2 * max_order_;
  multiindex_combination_.Init(list_total_num_coeffs_[limit],
                               list_total_num_coeffs_[limit]);

  for(int j = 0; j < list_total_num_coeffs_[limit]; j++) {

    // beta mapping
    const std::vector<short int> &beta_mapping = multiindex_mapping_[j];

    for(int k = 0; k < list_total_num_coeffs_[limit]; k++) {

      // alpha mapping
      const std::vector<short int> &alpha_mapping = multiindex_mapping_[k];

      // initialize the factor to 1
      multiindex_combination_.set(j, k, 1);

      for(int i = 0; i < dim_; i++) {
        multiindex_combination_.set
        (j, k, multiindex_combination_.get(j, k) *
         n_choose_k_.get(beta_mapping[i], alpha_mapping[i]));

        if(multiindex_combination_.get(j, k) == 0)
          break;
      }
    }
  }
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint& CartesianExpansionGlobal <ExpansionType>::
get_inv_multiindex_factorials() const {
  return inv_multiindex_factorials_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianExpansionGlobal <ExpansionType>::get_max_total_num_coeffs() const {
  return list_total_num_coeffs_[max_order_];
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const std::vector < short int > *CartesianExpansionGlobal <ExpansionType>::
get_lower_mapping_index()
const {
  return lower_mapping_index_.begin();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianExpansionGlobal <ExpansionType>::get_max_order() const {
  return max_order_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const std::vector < short int > &CartesianExpansionGlobal <
ExpansionType >::get_multiindex(int pos)
const {
  return multiindex_mapping_[pos];
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const std::vector < short int > *CartesianExpansionGlobal <
ExpansionType >::get_multiindex_mapping() const {
  return multiindex_mapping_.begin();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const core::table::DensePoint& CartesianExpansionGlobal <
ExpansionType >::get_neg_inv_multiindex_factorials() const {
  return neg_inv_multiindex_factorials_;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianExpansionGlobal <
ExpansionType >::get_n_choose_k(int n, int k) const {
  return n_choose_k_.get(n, (int) core::math::ClampNonNegative(k));
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianExpansionGlobal <
ExpansionType >::get_n_multichoose_k_by_pos(int n, int k) const {
  return multiindex_combination_.get(n, k);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianExpansionGlobal <
ExpansionType >::get_total_num_coeffs(int order) const {

  return list_total_num_coeffs_[order];
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
const std::vector < short int > *CartesianExpansionGlobal <
ExpansionType >::get_upper_mapping_index() const {

  return upper_mapping_index_.begin();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
int CartesianExpansionGlobal <
ExpansionType >::ComputeMultiindexPosition(
  const std::vector<short int> &multiindex) const {

  int dim = multiindex.size();
  int mapping_sum = 0;
  int index = 0;

  for(int j = 0; j < dim; j++) {

    // If any of the index is negative, then it does not exist!
    if(multiindex[j] < 0) {
      index = -1;
      break;
    }

    mapping_sum += multiindex[j];
  }
  if(index >= 0) {
    for(int j = 0; j < dim; j++) {
      index += (int) get_n_choose_k(mapping_sum + dim - j - 1, dim - j);
      mapping_sum -= multiindex[j];
    }
  }

  return index;
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianExpansionGlobal <
ExpansionType >::FarFieldEvaluationCost(
  int order) const {
  return pow(dim_, order + 1);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianExpansionGlobal <
ExpansionType >::FarFieldToLocalTranslationCost(int order) const {
  return pow(dim_, 2 * order + 1);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
double CartesianExpansionGlobal <
ExpansionType >::DirectLocalAccumulationCost(int order) const {
  return pow(dim_, order + 1);
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianExpansionGlobal <
ExpansionType >::Init(int max_order, int dim) {

  int p, k, t, tail, i, j;
  std::vector<int> heads;
  std::vector<int> cinds;

  // initialize max order and dimension
  dim_ = dim;
  max_order_ = max_order;

  // compute the list of total number of coefficients for p-th order expansion
  int limit = 2 * max_order + 1;
  list_total_num_coeffs_.resize(limit);
  for(p = 0; p < limit; p++) {
    list_total_num_coeffs_[p] =
      core::math::BinomialCoefficient<int>(p + dim, dim);
  }

  // compute factorials
  ComputeFactorials();

  // allocate space for inverse factorial and
  // negative inverse factorials and multiindex mapping and n_choose_k
  // and multiindex_combination precomputed factors
  inv_multiindex_factorials_.Init(list_total_num_coeffs_[limit - 1]);
  neg_inv_multiindex_factorials_.Init(list_total_num_coeffs_[limit - 1]);
  multiindex_mapping_.resize(list_total_num_coeffs_[limit - 1]);
  (multiindex_mapping_[0]).resize(dim_);
  for(j = 0; j < dim; j++) {
    (multiindex_mapping_[0])[j] = 0;
  }
  n_choose_k_.Init((limit - 1) + dim + 1, (limit - 1) + dim + 1);
  n_choose_k_.SetZero();

  // initialization of temporary variables for computation...
  heads.resize(dim + 1);
  cinds.resize(list_total_num_coeffs_[limit - 1]);

  for(i = 0; i < dim; i++) {
    heads[i] = 0;
  }
  heads[dim] = INT_MAX;
  cinds[0] = 0;

  // compute inverse factorial and negative inverse factorials and
  // multiindex mappings...
  inv_multiindex_factorials_[0] = 1.0;
  neg_inv_multiindex_factorials_[0] = 1.0;
  for(k = 1, t = 1, tail = 1; k <= 2 * max_order_; k++, tail = t) {
    for(i = 0; i < dim; i++) {
      int head = heads[i];
      heads[i] = t;
      for(j = head; j < tail; j++, t++) {
        cinds[t] = (j < heads[i + 1]) ? cinds[j] + 1 : 1;
        inv_multiindex_factorials_[t] =
          inv_multiindex_factorials_[j] / cinds[t];
        neg_inv_multiindex_factorials_[t] =
          -neg_inv_multiindex_factorials_[j] / cinds[t];

        // Copy using the STL vector copy operator.
        multiindex_mapping_[t] = multiindex_mapping_[j];
        (multiindex_mapping_[t])[i] = (multiindex_mapping_[t])[i] + 1;
      }
    }
  }

  // compute n choose k's
  for(j = 0; j <= 2 * max_order + dim; j++) {
    for(k = 0; k <= 2 * max_order + dim; k++) {
      n_choose_k_.set(j, k, core::math::BinomialCoefficient<double>(j, k));
    }
  }

  // initialize multiindex_combination matrix beta choose alpha
  ComputeMultiindexCombination();

  // compute the lower_mapping_index_ and the upper_mapping_index_
  // (see series_expansion_aux.h for explanation)
  ComputeLowerMappingIndex();
  ComputeUpperMappingIndex();
}

template<enum mlpack::series_expansion::CartesianExpansionType ExpansionType>
void CartesianExpansionGlobal <
ExpansionType >::Print(const char *name, FILE *stream) const {

  fprintf(stream, "----- CARTESIAN EXPANSION GLOBAL %s ------\n", name);
  fprintf(stream, "Max order: %d, dimension: %d\n", max_order_, dim_);

  fprintf(stream, "Multiindex mapping: ");
  for(int i = 0; i < multiindex_mapping_.size(); i++) {

    assert(
      ComputeMultiindexPosition(multiindex_mapping_[i]) == i);
    fprintf(stream, "( ");
    for(int j = 0; j < dim_; j++) {
      fprintf(stream, "%d ", multiindex_mapping_[i][j]);
    }
    fprintf(stream, "): %g %g ", inv_multiindex_factorials_[i],
            neg_inv_multiindex_factorials_[i]);
  }
  fprintf(stream, "\n");
}
}
}

#endif
