/** @file series_expansion_aux.h
 */

#ifndef SERIES_EXPANSION_AUX
#define SERIES_EXPANSION_AUX

#include <fastlib/fastlib.h>
#include <armadillo>

/**
 * Series expansion class.
 */
class SeriesExpansionAux {
  
 private:

  index_t dim_;

  index_t max_order_;

  arma::vec factorials_;

  std::vector<index_t> list_total_num_coeffs_;

  arma::vec inv_multiindex_factorials_;
  
  arma::vec neg_inv_multiindex_factorials_;

  arma::mat multiindex_combination_;

  std::vector< std::vector<index_t> > multiindex_mapping_;

  /** 
   * for each i-th multiindex m_i, store the positions of the j-th
   * multiindex mapping such that m_i - m_j >= 0 (the difference in
   * all coordinates is nonnegative).
   */
  std::vector< std::vector<index_t> > lower_mapping_index_;

  /** 
   * for each i-th multiindex m_i, store the positions of the j-th
   * multiindex mapping such that m_i - m_j <= 0 (the difference in
   * all coordinates is nonpositive).
   */
  std::vector< std::vector<index_t> > upper_mapping_index_;

  /** row index is for n, column index is for k */
  arma::mat n_choose_k_;

 public:

  void ComputeFactorials() {
    factorials_.set_size(2 * max_order_ + 1);

    factorials_[0] = 1;
    for(index_t t = 1; t < factorials_.n_elem; t++) {
      factorials_[t] = t * factorials_[t - 1];
    }
  }

  void ComputeLowerMappingIndex() {
    
    std::vector<index_t> diff;
    diff.reserve(dim_);

    index_t limit = 2 * max_order_;

    // initialize the index
    lower_mapping_index_.reserve(list_total_num_coeffs_[limit]);

    for(index_t i = 0; i < list_total_num_coeffs_[limit]; i++) {
      const std::vector<index_t> &outer_mapping = multiindex_mapping_[i];

      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
	const std::vector<index_t> &inner_mapping = multiindex_mapping_[j];
	index_t flag = 0;

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

    index_t limit = 2 * max_order_;
    multiindex_combination_.set_size(list_total_num_coeffs_[limit],
                                     list_total_num_coeffs_[limit]);

    for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
      
      // beta mapping
      const std::vector<index_t> &beta_mapping = multiindex_mapping_[j];
      
      for(index_t k = 0; k < list_total_num_coeffs_[limit]; k++) {
	
	// alpha mapping
	const std::vector<index_t> &alpha_mapping = multiindex_mapping_[k];
	
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
    
    index_t limit = 2 * max_order_;
    std::vector<index_t> diff;
    diff.reserve(dim_);
    
    // initialize the index
    upper_mapping_index_.reserve(list_total_num_coeffs_[limit]);
    
    for(index_t i = 0; i < list_total_num_coeffs_[limit]; i++) {
      const std::vector<index_t> &outer_mapping = multiindex_mapping_[i];
      
      for(index_t j = 0; j < list_total_num_coeffs_[limit]; j++) {
	const std::vector<index_t> &inner_mapping = multiindex_mapping_[j];
	index_t flag = 0;
	
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
  double factorial(index_t k) const { return factorials_[k]; }

  index_t get_dimension() const { return dim_; }

  index_t get_total_num_coeffs(index_t order) const;

  index_t get_max_total_num_coeffs() const;

  const arma::vec& get_inv_multiindex_factorials() const;

  const std::vector< std::vector<index_t> > & get_lower_mapping_index() const;

  index_t get_max_order() const;

  const std::vector< index_t > & get_multiindex(index_t pos) const;

  const std::vector< std::vector<index_t> > & get_multiindex_mapping() const;

  const arma::vec& get_neg_inv_multiindex_factorials() const;

  double get_n_choose_k(index_t n, index_t k) const;

  double get_n_multichoose_k_by_pos(index_t n, index_t k) const;

  const std::vector< std::vector<index_t> >& get_upper_mapping_index() const;

  // interesting functions

  /**
   * Computes the position of the given multiindex
   */
  index_t ComputeMultiindexPosition(const std::vector<index_t> &multiindex) const;

  /** @brief Computes the computational cost of evaluating a far-field
   *         expansion of order p at a single query point.
   */
  double FarFieldEvaluationCost(index_t order) const;

  /** @brief Computes the compuational cost of translating a far-field
   *         moment of order p into a local moment of the same order.
   */
  double FarFieldToLocalTranslationCost(index_t order) const;

  /** @brief Computes the computational cost of directly accumulating
   *         a single reference point into a local moment of order p.
   */
  double DirectLocalAccumulationCost(index_t order) const;

  /** 
   * Initialize the auxiliary object with precomputed quantities for
   * order up to max_order for the given dimensionality.
   */
  void Init(index_t max_order, index_t dim);

  /**
   * Print useful information about this object
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

};

#endif
