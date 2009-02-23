/** @file fourier_series_expansion_aux.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef FOURIER_SERIES_EXPANSION_AUX_H
#define FOURIER_SERIES_EXPANSION_AUX_H

#include "fastlib/fastlib.h"

/** @brief Fourier-series based expansion, essentially $O(p^D)$
 *         expansion.
 */
class FourierSeriesExpansionAux {

 public:

  int dim_;

  int max_order_;

  ArrayList<int> list_total_num_coeffs_;

  ArrayList< ArrayList<int> > multiindex_mapping_;

  OT_DEF_BASIC(FourierSeriesExpansionAux) {
    OT_MY_OBJECT(dim_);
    OT_MY_OBJECT(max_order_);
    OT_MY_OBJECT(list_total_num_coeffs_);
    OT_MY_OBJECT(multiindex_mapping_);
  }

 public:

  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const { 
    return list_total_num_coeffs_[order]; 
  }

  int get_max_total_num_coeffs() const { 
    return list_total_num_coeffs_[max_order_]; 
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
    return pow(2 * order + 1, dim_);
  }

  /** @brief Computes the compuational cost of translating a far-field
   *         moment of order p into a local moment of the same order.
   */
  double FarFieldToLocalTranslationCost(int order) const {
    return pow(2 * order + 1, dim_);
  }

  /** @brief Computes the computational cost of directly accumulating
   *         a single reference point into a local moment of order p.
   */
  double DirectLocalAccumulationCost(int order) const {
    return pow(2 * order + 1, dim_);
  }

  /** @brief Initialize the auxiliary object with precomputed
   *         quantities for order up to max_order for the given
   *         dimensionality.
   */
  void Init(int max_order, int dim) {

    // initialize max order and dimension
    dim_ = dim;
    max_order_ = max_order;
  
    // Compute the list of total number of coefficients for p-th order
    // expansion
    int limit = max_order_;
    list_total_num_coeffs_.Init(limit + 1);
    for(index_t p = 0; p <= limit; p++) {
      list_total_num_coeffs_[p] = (int) pow(2 * p + 1, dim);
    }

    // Compute the multiindex mappings...
    multiindex_mapping_.Init(list_total_num_coeffs_[limit]);
    (multiindex_mapping_[0]).Init(dim_);
    for(index_t j = 0; j < dim; j++) {
      (multiindex_mapping_[0])[j] = -max_order;
    }
    printf("Hi %d\n", list_total_num_coeffs_[limit]);
    if(max_order > 0) {
      index_t boundary, i, k, step;

      for(boundary = list_total_num_coeffs_[limit], k = 0, 
	    step = list_total_num_coeffs_[limit] / (2 * limit + 1);
	  step >= 1; step /= (2 * limit + 1), 
	    boundary /= (2 * limit + 1), k++) {

	for(i = 0; i < list_total_num_coeffs_[limit]; ) {
	  int inner_limit = i + boundary;
	  int div = 1;
	  
	  i += step;
	  
	  for( ; i < inner_limit; i += step) {

	    div++;

	    // copy multiindex from old to the new position
	    multiindex_mapping_[i].InitCopy(multiindex_mapping_[i - step]);
	    (multiindex_mapping_[i])[k] = (multiindex_mapping_[i])[k] + 1;
	  }
	}
      }
    }
  }

  /**
   * Print useful information about this object
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const {
    fprintf(stream, "----- SERIESEXPANSIONAUX %s ------\n", name);
    fprintf(stream, "Max order: %d, dimension: %d\n", max_order_, dim_);

    fprintf(stream, "Multiindex mapping: ");
    for (index_t i = 0; i < multiindex_mapping_.size(); i++) {
      fprintf(stream, "( ");
      for(index_t j = 0; j < dim_; j++) {
	fprintf(stream, "%d ", multiindex_mapping_[i][j]);
      }
      fprintf(stream, ") ");
    }
    fprintf(stream, "\n");
  }

};

#endif
