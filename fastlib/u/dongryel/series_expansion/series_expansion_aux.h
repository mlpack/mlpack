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

  /** row index is for n, column index is for k */
  Matrix n_choose_k_;

  /** compute n choose k: this might be moved later to FastLib core... */
  int nchoosek(int n, int k) {
    int n_k = n - k;
    int nchsk = 1;
    int i;
    
    if(k > n)
      return 0;
    
    if(k < n_k) {
      k = n_k;
      n_k = n - k;
    }
    
    for(i = 1; i <= n_k; i++) {
      nchsk *= (++k);
      nchsk /= i;
    }
    return nchsk;
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
