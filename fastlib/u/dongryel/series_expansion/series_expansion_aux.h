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

  /** row index is for n, column index is for k */
  Matrix n_choose_k_;

  void ComputeLowerMappingIndex() {
    
    ArrayList<int> diff;
    diff.Init(dim_);

    // initialize the index
    lower_mapping_index_.Init(list_total_num_coeffs_[max_order_]);

    for(index_t i = 0; i < list_total_num_coeffs_[max_order_]; i++) {
      ArrayList<int> outer_mapping = multiindex_mapping_[i];
      lower_mapping_index_[i].Init();

      for(index_t j = 0; j <= i; j++) {
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

 public:

  // construtor/destructor
  SeriesExpansionAux() {}

  ~SeriesExpansionAux() {}

  // getters and setters
  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const;

  int get_max_total_num_coeffs() const;

  const Vector& get_inv_multiindex_factorials() const;

  const ArrayList< int > * get_lower_mapping_index() const;

  const ArrayList< int > & get_multiindex(int pos) const;

  const ArrayList< int > * get_multiindex_mapping() const;

  const Vector& get_neg_inv_multiindex_factorials() const;

  double get_n_choose_k(int n, int k) const;

  double get_n_multichoose_k_by_pos(int n, int k) const;

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
