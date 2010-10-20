/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/** @file mult_series_expansion_aux.h
 */

#ifndef MULT_SERIES_EXPANSION_AUX_H
#define MULT_SERIES_EXPANSION_AUX_H

#include "fastlib/fastlib.h"

/**
 * Series expansion class for multiplicative kernel functions
 * Precomputes constants for O(p^D) expansions.
 */
class MultSeriesExpansionAux {

 public:

  int dim_;

  int max_order_;

  Vector factorials_;

  ArrayList<int> list_total_num_coeffs_;

  Vector inv_multiindex_factorials_;

  Vector neg_inv_multiindex_factorials_;

  Matrix multiindex_combination_;

  ArrayList< ArrayList<int> > multiindex_mapping_;

  /**
   * for each i-th multiindex m_i, store the positions of the j-th
   * multiindex mapping such that m_i - m_j >= 0 (the difference in
   * all coordinates is nonnegative).
   */
  ArrayList< ArrayList<int> > lower_mapping_index_;

  /**
   * for each i-th multiindex m_i, store the positions of the j-th
   * multiindex mapping such that m_i - m_j <= 0 (the difference in
   * all coordinates is nonpositive).
   */
  ArrayList< ArrayList<int> > upper_mapping_index_;

  /** row index is for n, column index is for k */
  Matrix n_choose_k_;

  /**
   * For each i-th order, store the positions of the coefficient
   * array to traverse.
   */
  ArrayList< ArrayList<int> > traversal_mapping_;

/*  OT_DEF_BASIC(MultSeriesExpansionAux) {
    OT_MY_OBJECT(dim_);
    OT_MY_OBJECT(max_order_);
    OT_MY_OBJECT(factorials_);
    OT_MY_OBJECT(list_total_num_coeffs_);
    OT_MY_OBJECT(inv_multiindex_factorials_);
    OT_MY_OBJECT(neg_inv_multiindex_factorials_);
    OT_MY_OBJECT(multiindex_combination_);
    OT_MY_OBJECT(multiindex_mapping_);
    OT_MY_OBJECT(lower_mapping_index_);
    OT_MY_OBJECT(upper_mapping_index_);
    OT_MY_OBJECT(n_choose_k_);
    OT_MY_OBJECT(traversal_mapping_);
  }
*/

  friend class boost::serialization::access; // Should be removed later

  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & dim_;
    ar & max_order_;
    //Vector to be replaced with std::vector
    //ar & factorials_;
    //ArrayList to be replaced with std::vector
    //ar & list_total_num_coeffs_;
    //Vector to be replaced with std::vector
    //ar & inv_multiindex_factorials_;
    //ar & neg_inv_multiindex_factorials_;
    // Matrix to be replaced with sparse armadillo matrices
    //ar & multiindex_combination_;
    //ar & multiindex_mapping_;
    //ar & lower_mapping_index_;
    //ar & upper_mapping_index_;
    //ar & n_choose_k_;
    //ar & traversal_mapping_;
  }

 public:

  void ComputeFactorials() {
    factorials_.Init(2 * max_order_ + 1);

    factorials_[0] = 1;
    for(index_t t = 1; t < factorials_.length(); t++) {
      factorials_[t] = t * factorials_[t - 1];
    }
  }

  void ComputeTraversalMapping() {

    // initialize the index
    int limit = 2 * max_order_;
    traversal_mapping_.Init(limit + 1);

    for(index_t i = 0; i <= max_order_; i++) {

      traversal_mapping_[i].Init();

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
	
        const ArrayList<int> &mapping = multiindex_mapping_[j];
        int flag = 0;

        for(index_t d = 0; d < dim_; d++) {
          if(mapping[d] > i) {
            flag = 1;
            break;
          }
        }

        if(flag == 0) {
          (traversal_mapping_[i]).PushBackCopy(j);
        }
      } // end of j-loop
    } // end of i-loop
  }

  void ComputeLowerMappingIndex() {

    ArrayList<int> diff;
    diff.Init(dim_);

    // initialize the index
    int limit = 2 * max_order_;
    lower_mapping_index_.Init(list_total_num_coeffs_[limit]);

    for(index_t i = 0; i < list_total_num_coeffs_[limit]; i++) {
      const ArrayList<int> &outer_mapping = multiindex_mapping_[i];
      lower_mapping_index_[i].Init();

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
        const ArrayList<int> &inner_mapping = multiindex_mapping_[j];
        int flag = 0;

        for(index_t d = 0; d < dim_; d++) {
          diff[d] = outer_mapping[d] - inner_mapping[d];

          if(diff[d] < 0) {
            flag = 1;
            break;
          }
        }

        if(flag == 0) {
          (lower_mapping_index_[i]).PushBackCopy(j);
        }
      } // end of j-loop
    } // end of i-loop
  }

  void ComputeMultiindexCombination() {

    int limit = 2 * max_order_;
    multiindex_combination_.Init(list_total_num_coeffs_[limit],
                                 list_total_num_coeffs_[limit]);

    for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {

      // beta mapping
      const ArrayList<int> &beta_mapping = multiindex_mapping_[j];

      for(index_t k = 0; k < list_total_num_coeffs_[limit]; k++) {

        // alpha mapping
        const ArrayList<int> &alpha_mapping = multiindex_mapping_[k];

        // initialize the factor to 1
        multiindex_combination_.set(j, k, 1);

        for(index_t i = 0; i < dim_; i++) {
          multiindex_combination_.set
            (j, k, multiindex_combination_.get(j, k) *
             n_choose_k_.get(beta_mapping[i], alpha_mapping[i]));

          if(multiindex_combination_.get(j, k) == 0)
            break;
        }
      }
    }
  }

  void ComputeUpperMappingIndex() {

    ArrayList<int> diff;
    diff.Init(dim_);

    // initialize the index
    int limit = 2 * max_order_;
    upper_mapping_index_.Init(list_total_num_coeffs_[limit]);

    for(index_t i = 0; i < list_total_num_coeffs_[limit]; i++) {
      const ArrayList<int> &outer_mapping = multiindex_mapping_[i];
      upper_mapping_index_[i].Init();

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
        const ArrayList<int> &inner_mapping = multiindex_mapping_[j];
        int flag = 0;

        for(index_t d = 0; d < dim_; d++) {
          diff[d] = inner_mapping[d] - outer_mapping[d];

          if(diff[d] < 0) {
            flag = 1;
            break;
          }
        }

        if(flag == 0) {
          (upper_mapping_index_[i]).PushBackCopy(j);
        }
      } // end of j-loop
    } // end of i-loop
  }

  // getters and setters
  double factorial(int k) const { return factorials_[k]; }

  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const { 
    return list_total_num_coeffs_[order]; 
  }

  int get_max_total_num_coeffs() const { 
    return list_total_num_coeffs_[max_order_]; 
  }

  const Vector& get_inv_multiindex_factorials() const {
    return inv_multiindex_factorials_;
  }

  const ArrayList< int > * get_lower_mapping_index() const {
    return lower_mapping_index_.begin();
  }

  int get_max_order() const {
    return max_order_;
  }

  const ArrayList< int > & get_multiindex(int pos) const {
    return multiindex_mapping_[pos];
  }

  const ArrayList< int > * get_multiindex_mapping() const {
    return multiindex_mapping_.begin();
  }

  const Vector& get_neg_inv_multiindex_factorials() const {
    return neg_inv_multiindex_factorials_;
  }

  double get_n_choose_k(int n, int k) const {
    return n_choose_k_.get(n, (int) math::ClampNonNegative(k));
  }

  double get_n_multichoose_k_by_pos(int n, int k) const {
    return multiindex_combination_.get(n, k);
  }

  const ArrayList< int > * get_upper_mapping_index() const {
    return upper_mapping_index_.begin();
  }

  // interesting functions

  /**
   * Computes the position of the given multiindex
   */
  int ComputeMultiindexPosition(const ArrayList<int> &multiindex) const {
    int index = 0;
    
    // using Horner's rule
    for(index_t i = 0; i < dim_; i++) {
      index *= (2 * max_order_ + 1);
      index += multiindex[i];
    }
    return index;
  }

  /** @brief Computes the computational cost of evaluating a far-field
   *         expansion of order p at a single query point.
   */
  double FarFieldEvaluationCost(int order) const {
    return pow(order + 1, dim_);
  }

  /** @brief Computes the compuational cost of translating a far-field
   *         moment of order p into a local moment of the same order.
   */
  double FarFieldToLocalTranslationCost(int order) const {
    return pow(order + 1, 2 * dim_);
  }

  /** @brief Computes the computational cost of directly accumulating
   *         a single reference point into a local moment of order p.
   */
  double DirectLocalAccumulationCost(int order) const {
    return pow(order + 1, dim_);
  }

  /** @brief Initialize the auxiliary object with precomputed
   *         quantities for order up to max_order for the given
   *         dimensionality.
   */
  void Init(int max_order, int dim) {

    // initialize max order and dimension
    dim_ = dim;
    max_order_ = max_order;
  
    // compute the list of total number of coefficients for p-th order 
    // expansion
    int limit = 2 * max_order_;
    list_total_num_coeffs_.Init(limit + 1);
    list_total_num_coeffs_[0] = 1;
    for(index_t p = 1; p <= limit; p++) {
      list_total_num_coeffs_[p] = (int) pow(p + 1, dim);
    }

    // compute factorials
    ComputeFactorials();

    // allocate space for inverse factorial and 
    // negative inverse factorials and multiindex mapping and n_choose_k 
    // and multiindex_combination precomputed factors
    inv_multiindex_factorials_.Init(list_total_num_coeffs_[limit]);  
    neg_inv_multiindex_factorials_.Init(list_total_num_coeffs_[limit]);
    multiindex_mapping_.Init(list_total_num_coeffs_[limit]);
    (multiindex_mapping_[0]).Init(dim_);
    for(index_t j = 0; j < dim; j++) {
      (multiindex_mapping_[0])[j] = 0;
    }
    n_choose_k_.Init(dim * (limit + 1), dim * (limit + 1));
    n_choose_k_.SetZero();

    // compute inverse factorial and negative inverse factorials and
    // multiindex mappings...
    inv_multiindex_factorials_[0] = 1.0;
    neg_inv_multiindex_factorials_[0] = 1.0;
    if(max_order > 0) {
      index_t boundary, i, k, step;

      for(boundary = list_total_num_coeffs_[limit], k = 0, 
	    step = list_total_num_coeffs_[limit] / (limit + 1);
	  step >= 1; step /= (limit + 1), 
	    boundary /= (limit + 1), k++) {

	for(i = 0; i < list_total_num_coeffs_[limit]; ) {
	  int inner_limit = i + boundary;
	  int div = 1;
	  
	  i += step;
	  
	  for( ; i < inner_limit; i += step) {
	    
	    inv_multiindex_factorials_[i] = 
	      inv_multiindex_factorials_[i - step] / div;
	    neg_inv_multiindex_factorials_[i] =
	      -neg_inv_multiindex_factorials_[i - step] / div;
	    div++;

	    // copy multiindex from old to the new position
	    multiindex_mapping_[i].InitCopy(multiindex_mapping_[i - step]);
	    (multiindex_mapping_[i])[k] = (multiindex_mapping_[i])[k] + 1;
	  }
	}
      }
    }

    // compute n choose k's
    for(index_t j = 0; j < n_choose_k_.n_rows(); j++) {
      for(index_t k = 0; k < n_choose_k_.n_cols(); k++) {
	n_choose_k_.set(j, k, math::BinomialCoefficient(j, k));
      }
    }

    // initialize multiindex_combination matrix beta choose alpha
    ComputeMultiindexCombination();

    // compute the lower_mapping_index_ and the upper_mapping_index_
    // (see series_expansion_aux.h for explanation)
    ComputeLowerMappingIndex();
    ComputeUpperMappingIndex();

    // compute traversal mapping
    ComputeTraversalMapping();
  }

  /**
   * Print useful information about this object
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const {
    fprintf(stream, "----- SERIESEXPANSIONAUX %s ------\n", name);
    fprintf(stream, "Max order: %d, dimension: %d\n", max_order_, dim_);

    fprintf(stream, "Multiindex mapping: ");
    for (index_t i = 0; i < multiindex_mapping_.size(); i++) {

      DEBUG_ASSERT_MSG(ComputeMultiindexPosition(multiindex_mapping_[i]) == i,
		       "REIMPLEMENT ComputeMultiindexPosition function!");
      fprintf(stream, "( ");
      for(index_t j = 0; j < dim_; j++) {
	fprintf(stream, "%d ", multiindex_mapping_[i][j]);
      }
      fprintf(stream, "): %g %g ", inv_multiindex_factorials_[i],
	      neg_inv_multiindex_factorials_[i]);
    }
    fprintf(stream, "\n");
  }

};

#endif
