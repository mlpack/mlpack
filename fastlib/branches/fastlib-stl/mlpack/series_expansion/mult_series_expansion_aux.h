/** @file mult_series_expansion_aux.h
 */

#ifndef MULT_SERIES_EXPANSION_AUX_H
#define MULT_SERIES_EXPANSION_AUX_H

#include <fastlib/fastlib.h>
#include <armadillo>

/**
 * Series expansion class for multiplicative kernel functions
 * Precomputes constants for O(p^D) expansions.
 */
class MultSeriesExpansionAux {

 public:

  int dim_;

  int max_order_;

  arma::vec factorials_;

  std::vector<int> list_total_num_coeffs_;

  arma::vec inv_multiindex_factorials_;

  arma::vec neg_inv_multiindex_factorials_;

  arma::mat multiindex_combination_;

  std::vector< std::vector<short int> > multiindex_mapping_;

  /**
   * for each i-th multiindex m_i, store the positions of the j-th
   * multiindex mapping such that m_i - m_j >= 0 (the difference in
   * all coordinates is nonnegative).
   */
  std::vector< std::vector<short int> > lower_mapping_index_;

  /**
   * for each i-th multiindex m_i, store the positions of the j-th
   * multiindex mapping such that m_i - m_j <= 0 (the difference in
   * all coordinates is nonpositive).
   */
  std::vector< std::vector<short int> > upper_mapping_index_;

  /** row index is for n, column index is for k */
  arma::mat n_choose_k_;

  /**
   * For each i-th order, store the positions of the coefficient
   * array to traverse.
   */
  std::vector< std::vector<short int> > traversal_mapping_;

 public:

  void ComputeFactorials() {
    factorials_.set_size(2 * max_order_ + 1);

    factorials_[0] = 1;
    for(index_t t = 1; t < factorials_.n_elem; t++) {
      factorials_[t] = t * factorials_[t - 1];
    }
  }

  void ComputeTraversalMapping() {

    // initialize the index
    int limit = 2 * max_order_;
    traversal_mapping_.reserve(limit + 1);

    for(index_t i = 0; i <= max_order_; i++) {

      traversal_mapping_[i].clear();

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
	
        const std::vector<short int>& mapping = multiindex_mapping_[j];
        int flag = 0;

        for(index_t d = 0; d < dim_; d++) {
          if(mapping[d] > i) {
            flag = 1;
            break;
          }
        }

        if(flag == 0) {
          (traversal_mapping_[i]).push_back(j);
        }
      } // end of j-loop
    } // end of i-loop
  }

  void ComputeLowerMappingIndex() {

    std::vector<short int> diff;
    diff.reserve(dim_);

    // initialize the index
    int limit = 2 * max_order_;
    lower_mapping_index_.reserve(list_total_num_coeffs_[limit]);

    for(index_t i = 0; i < list_total_num_coeffs_[limit]; i++) {
      const std::vector<short int>& outer_mapping = multiindex_mapping_[i];
      lower_mapping_index_[i].clear();

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
        const std::vector<short int>& inner_mapping = multiindex_mapping_[j];
        int flag = 0;

        for(index_t d = 0; d < dim_; d++) {
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

  void ComputeMultiindexCombination() {

    int limit = 2 * max_order_;
    multiindex_combination_.set_size(list_total_num_coeffs_[limit],
                                     list_total_num_coeffs_[limit]);

    for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {

      // beta mapping
      const std::vector<short int>& beta_mapping = multiindex_mapping_[j];

      for(index_t k = 0; k < list_total_num_coeffs_[limit]; k++) {

        // alpha mapping
        const std::vector<short int>& alpha_mapping = multiindex_mapping_[k];

        // initialize the factor to 1
        multiindex_combination_(j, k) = 1;

        for(index_t i = 0; i < dim_; i++) {
          multiindex_combination_(j, k) *=
             n_choose_k_(beta_mapping[i], alpha_mapping[i]);

          if(multiindex_combination_(j, k) == 0)
            break;
        }
      }
    }
  }

  void ComputeUpperMappingIndex() {

    std::vector<short int> diff;
    diff.reserve(dim_);

    // initialize the index
    int limit = 2 * max_order_;
    upper_mapping_index_.reserve(list_total_num_coeffs_[limit]);

    for(index_t i = 0; i < list_total_num_coeffs_[limit]; i++) {
      const std::vector<short int>& outer_mapping = multiindex_mapping_[i];
      upper_mapping_index_[i].clear();

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
        const std::vector<short int>& inner_mapping = multiindex_mapping_[j];
        int flag = 0;

        for(index_t d = 0; d < dim_; d++) {
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

  // getters and setters
  double factorial(int k) const { return factorials_[k]; }

  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const { 
    return list_total_num_coeffs_[order]; 
  }

  int get_max_total_num_coeffs() const { 
    return list_total_num_coeffs_[max_order_]; 
  }

  const arma::vec& get_inv_multiindex_factorials() const {
    return inv_multiindex_factorials_;
  }

  const std::vector<std::vector<short int> >& get_lower_mapping_index() const {
    return lower_mapping_index_;
  }

  int get_max_order() const {
    return max_order_;
  }

  const std::vector<short int>& get_multiindex(int pos) const {
    return multiindex_mapping_[pos];
  }

  const std::vector<std::vector<short int> >& get_multiindex_mapping() const {
    return multiindex_mapping_;
  }

  const arma::vec& get_neg_inv_multiindex_factorials() const {
    return neg_inv_multiindex_factorials_;
  }

  double get_n_choose_k(int n, int k) const {
    return n_choose_k_(n, (int) math::ClampNonNegative(k));
  }

  double get_n_multichoose_k_by_pos(int n, int k) const {
    return multiindex_combination_(n, k);
  }

  const std::vector<std::vector<short int> >& get_upper_mapping_index() const {
    return upper_mapping_index_;
  }

  // interesting functions

  /**
   * Computes the position of the given multiindex
   */
  int ComputeMultiindexPosition(const std::vector<short int>& multiindex) const {
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
    list_total_num_coeffs_.reserve(limit + 1);
    list_total_num_coeffs_[0] = 1;
    for(index_t p = 1; p <= limit; p++) {
      list_total_num_coeffs_[p] = (int) pow(p + 1, dim);
    }

    // compute factorials
    ComputeFactorials();

    // allocate space for inverse factorial and 
    // negative inverse factorials and multiindex mapping and n_choose_k 
    // and multiindex_combination precomputed factors
    inv_multiindex_factorials_.set_size(list_total_num_coeffs_[limit]);  
    neg_inv_multiindex_factorials_.set_size(list_total_num_coeffs_[limit]);
    multiindex_mapping_.reserve(list_total_num_coeffs_[limit]);
    (multiindex_mapping_[0]).reserve(dim_);
    for(index_t j = 0; j < dim; j++) {
      (multiindex_mapping_[0])[j] = 0;
    }
    n_choose_k_.zeros(dim * (limit + 1), dim * (limit + 1));

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
	    multiindex_mapping_[i] = multiindex_mapping_[i - step];
	    (multiindex_mapping_[i])[k] = (multiindex_mapping_[i])[k] + 1;
	  }
	}
      }
    }

    // compute n choose k's
    for(index_t j = 0; j < n_choose_k_.n_rows; j++) {
      for(index_t k = 0; k < n_choose_k_.n_cols; k++) {
	n_choose_k_(j, k) = math::BinomialCoefficient(j, k);
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
