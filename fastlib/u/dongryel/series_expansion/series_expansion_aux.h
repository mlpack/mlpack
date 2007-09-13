/** @file series_expansion_aux.h
 */

#ifndef SERIES_EXPANSION_AUX
#define SERIES_EXPANSION_AUX

#include "fastlib/fastlib.h"

/**
 * Series expansion class.
 */
class SeriesExpansionAux {
  FORBID_COPY(SeriesExpansionAux);
  
 private:

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

  void ComputeFactorials() {
    factorials_.Init(max_order_ + 1);

    factorials_[0] = 1;
    for(index_t t = 1; t <= max_order_; t++) {
      factorials_[t] = t * factorials_[t - 1];
    }
  }

  void ComputeLowerMappingIndex() {
    
    ArrayList<int> diff;
    diff.Init(dim_);

    // initialize the index
    lower_mapping_index_.Init(list_total_num_coeffs_[max_order_]);

    for(index_t i = 0; i < list_total_num_coeffs_[max_order_]; i++) {
      ArrayList<int> outer_mapping = multiindex_mapping_[i];
      lower_mapping_index_[i].Init();

      for(index_t j = 0; j < list_total_num_coeffs_[max_order_]; j++) {
	ArrayList<int> inner_mapping = multiindex_mapping_[j];
	int flag = 0;

	for(index_t d = 0; d < dim_; d++) {
	  diff[d] = outer_mapping[d] - inner_mapping[d];
	  
	  if(diff[d] < 0) {
	    flag = 1;
	    break;
	  }
	}
	
	if(flag == 0) {
	  (lower_mapping_index_[i]).AddBackItem(j);
	}
      } // end of j-loop
    } // end of i-loop
  }

  void ComputeMultiindexCombination() {

    multiindex_combination_.Init(list_total_num_coeffs_[max_order_],
				 list_total_num_coeffs_[max_order_]);

    for(index_t j = 0; j < list_total_num_coeffs_[max_order_]; j++) {
      
      // beta mapping
      ArrayList<int> beta_mapping = multiindex_mapping_[j];
      
      for(index_t k = 0; k < list_total_num_coeffs_[max_order_]; k++) {
	
	// alpha mapping
	ArrayList<int> alpha_mapping = multiindex_mapping_[k];
	
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
    upper_mapping_index_.Init(list_total_num_coeffs_[max_order_]);
    
    for(index_t i = 0; i < list_total_num_coeffs_[max_order_]; i++) {
      ArrayList<int> outer_mapping = multiindex_mapping_[i];
      upper_mapping_index_[i].Init();
      
      for(index_t j = 0; j < list_total_num_coeffs_[max_order_]; j++) {
	ArrayList<int> inner_mapping = multiindex_mapping_[j];
	int flag = 0;
	
	for(index_t d = 0; d < dim_; d++) {
	  diff[d] = inner_mapping[d] - outer_mapping[d];
	  
	  if(diff[d] < 0) {
	    flag = 1;
	    break;
	  }
	}
	
	if(flag == 0) {
	  (upper_mapping_index_[i]).AddBackItem(j);
	}
      } // end of j-loop
    } // end of i-loop
  }

 public:

  // construtor/destructor
  SeriesExpansionAux() {}

  ~SeriesExpansionAux() {}

  // getters and setters
  double factorial(int k) { return factorials_[k]; }

  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const;

  int get_max_total_num_coeffs() const;

  const Vector& get_inv_multiindex_factorials() const;

  const ArrayList< int > * get_lower_mapping_index() const;

  int get_max_order() const;

  const ArrayList< int > & get_multiindex(int pos) const;

  const ArrayList< int > * get_multiindex_mapping() const;

  const Vector& get_neg_inv_multiindex_factorials() const;

  double get_n_choose_k(int n, int k) const;

  double get_n_multichoose_k_by_pos(int n, int k) const;

  const ArrayList< int > * get_upper_mapping_index() const;

  // interesting functions

  /**
   * Computes the position of the given multiindex
   */
  int ComputeMultiindexPosition(const ArrayList<int> &multiindex) const;

  /** 
   * Initialize the auxiliary object with precomputed quantities for
   * order up to max_order for the given dimensionality.
   */
  void Init(int max_order, int dim);

  /**
   * Print useful information about this object
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

};

#endif
