/** @file fourier_series_expansion_aux.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef FOURIER_SERIES_EXPANSION_AUX_H
#define FOURIER_SERIES_EXPANSION_AUX_H

#include <fastlib/fastlib.h>
#include <iostream>

/** @brief Fourier-series based expansion, essentially $O(p^D)$
 *         expansion.
 */
template<typename T = double>
class FourierSeriesExpansionAux {

 public:

  int dim_;

  int max_order_;

  double integral_truncation_limit_;

  std::vector<short int> list_total_num_coeffs_;

  arma::Col<T> precomputed_constants_;

  T final_scaling_factor_;

  std::vector< std::vector<short int> > multiindex_mapping_;

 public:

  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const { 
    return list_total_num_coeffs_[order]; 
  }

  int get_max_total_num_coeffs() const { 
    return list_total_num_coeffs_[max_order_]; 
  }

  double integral_truncation_limit() const {
    return integral_truncation_limit_;
  }

  int get_max_order() const {
    return max_order_;
  }

  void set_integral_truncation_limit(double integral_truncation_limit_in) {
    integral_truncation_limit_ = integral_truncation_limit_in;
  }

  const std::vector<short int>& get_multiindex(int pos) const {
    return multiindex_mapping_[pos];
  }

  const std::vector<std::vector<short int> >& get_multiindex_mapping() const {
    return multiindex_mapping_.begin();
  }

  // interesting functions

  /**
   * Computes the position of the given multiindex
   */
  int ComputeMultiindexPosition(const std::vector<short int> &multiindex) const {
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

    // Set the integral truncation limit. This perhaps should be set
    // to a user-defined level.
    integral_truncation_limit_ = 10;

    // initialize max order and dimension
    dim_ = dim;
    max_order_ = max_order;
  
    // Compute the list of total number of coefficients for p-th order
    // expansion
    int limit = max_order_;
    list_total_num_coeffs_.reserve(limit + 1);
    for(index_t p = 0; p <= limit; p++) {
      list_total_num_coeffs_[p] = (int) pow(2 * p + 1, dim);
    }

    // Compute the multiindex mappings...
    multiindex_mapping_.reserve(list_total_num_coeffs_[limit]);
    multiindex_mapping_[0].reserve(dim_);
    for(index_t j = 0; j < dim; j++) {
      (multiindex_mapping_[0])[j] = -max_order;
    }
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
	    multiindex_mapping_[i] = multiindex_mapping_[i - step];
	    (multiindex_mapping_[i])[k] = (multiindex_mapping_[i])[k] + 1;
	  }
	}
      }
    }

    // Now loop over each multi-index, and precompute the constants.
    precomputed_constants_.set_size(multiindex_mapping_.size());
    final_scaling_factor_ = 
      pow(integral_truncation_limit_ / 
	  (2.0 * max_order_ * sqrt(2.0 * acos(0))), dim_);
    for(index_t j = 0; j < multiindex_mapping_.size(); j++) {
      const std::vector<short int> &mapping = multiindex_mapping_[j];

      // The squared length of the multiindex mapping.
      double squared_length_mapping = 0;
      for(index_t k = 0; k < dim_; k++) {
	squared_length_mapping += pow(mapping[k], 2.0);
      }
      
      precomputed_constants_[j] = 
	exp(-squared_length_mapping * pow(integral_truncation_limit_, 2.0) /
	    (4 * pow(max_order_, 2.0)));
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

  /** @brief The common translation operator used for far-to-far,
   *         far-to-local, and local-to-local translation.
   */
  template<typename SourceExpansion, typename DestinationExpansion>
  void TranslationOperator
  (const SourceExpansion &source_expansion,
   DestinationExpansion &destination_expansion, int truncation_order) const {
    
    // Get the coefficients to be translated and its center of
    // expansion.
    typedef typename SourceExpansion::data_type T;
    const arma::Col<T> &coeffs_to_be_translated = source_expansion.get_coeffs();    
    const arma::Col<T> &source_center = source_expansion.get_center();
    const arma::Col<T> &destination_center = destination_expansion.get_center();
    arma::Col<T> &coeffs_destination = destination_expansion.get_coeffs();
    
    // Loop over each coefficient.
    index_t num_coefficients = list_total_num_coeffs_[truncation_order];
    for(index_t i = 0; i < num_coefficients; i++) {
      
      // Get the current multi-index.
      const std::vector<short int> &mapping = get_multiindex(i);
      
      // Dot product between the multiindex and the center
      // differences.
      double dot_product = 0;
      for(index_t j = 0; j < mapping.size(); j++) {
	dot_product += mapping[j] * (destination_center[j] - source_center[j]);
      }
      
      // For each coefficient, scale it and add to the current one.
      double trig_argument = integral_truncation_limit() * dot_product /
	(get_max_order() * 
	 sqrt(2 * (source_expansion.kernel_aux_)->bandwidth_sq()));
      std::complex<T> factor(cos(trig_argument), sin(trig_argument));
      std::complex<T> new_coefficient = coeffs_to_be_translated.get(i) * 
	factor;
      
      coeffs_destination(i) = new_coefficient;
    }    
  }

  /** @brief The common series expansion evaluation for far-field and
   *         local expansion based on Fourier expansion.
   */
  template<typename TExpansion>
  typename TExpansion::data_type EvaluationOperator
  (const TExpansion &expansion, const typename TExpansion::data_type *x_q, 
   int order) const {
    
    typedef typename TExpansion::data_type T;
    std::complex<typename TExpansion::data_type> result = 0;
    index_t num_coefficients = list_total_num_coeffs_[order];

    // The coefficients to be evaluated.
    const ComplexVector<typename TExpansion::data_type> &coeffs = 
      expansion.get_coeffs();
    const arma::Col<typename TExpansion::data_type> &source_center =
      expansion.get_center();
    
    for(index_t i = 0; i < num_coefficients; i++) {
      
      // Get the current multi-index.
      const std::vector<short int> &mapping = get_multiindex(i);
      
      // Dot product between the multiindex and the center
      // differences.
      double dot_product = 0;
      for(index_t j = 0; j < mapping.size(); j++) {
	dot_product += mapping[j] * (x_q[j] - source_center[j]);
      }

      // For each coefficient, scale it and add to the current one.
      double trig_argument = integral_truncation_limit() * dot_product /
	(get_max_order() * 
	 sqrt(2 * (expansion.ka_->kernel_.bandwidth_sq())));
      std::complex<T> factor(cos(trig_argument), sin(trig_argument));
      std::complex<T> new_coefficient = coeffs.get(i) * factor;

      result += precomputed_constants_[i] * new_coefficient;
    }

    // Scale the result then return.
    result *= final_scaling_factor_;
    return result.real();
  }

};

#endif
