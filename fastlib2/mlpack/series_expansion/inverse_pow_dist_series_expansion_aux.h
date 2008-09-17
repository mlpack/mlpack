/** @file inverse_pow_dist_series_expansion_aux.h
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef INVERSE_POW_DIST_SERIES_EXPANSION_AUX
#define INVERSE_POW_DIST_SERIES_EXPANSION_AUX

#include "fastlib/fastlib.h"

/** @brief Series expansion class for the inverse distance power
 *         functions.
 */
class InversePowDistSeriesExpansionAux {
  
 private:

  /** @brief The inverse power $1 / r^{\lambda}$ for which this object
   *         is relevant.
   */
  double lambda_;

  int dim_;

  int max_order_;

  /** @brief These are the $(-1)^n (n C a) (a C b) / (2^a n!)$
   *         constants. $n$ index the most outer ArrayList, and (a, b)
   *         are the indices on each matrix.
   */
  ArrayList<Matrix> precomputed_constants_;
  
  OT_DEF_BASIC(InversePowDistSeriesExpansionAux) {
    OT_MY_OBJECT(dim_);
    OT_MY_OBJECT(max_order_);
    OT_MY_OBJECT(precomputed_constants_);
  }

 public:

  void ComputeConstants() {
    
    double n_factorial = 1.0;
    precomputed_constants_.Init(max_order_);

    for(index_t n = 0; n < precomputed_constants_.size(); n++) {
      
      // Allocate $(n + 1)$ by $(n + 1)$ matrix.
      precomputed_constants_[n].Init(n + 1, n + 1);

      // The reference to the matrix.
      Matrix &n_th_order_matrix = precomputed_constants_[n];

      double two_raised_to_a = 1.0;
      for(index_t a = 0; a <= n; a++) {

	for(index_t b = 0; b <= a; b++) {
	  n_th_order_matrix.set(a, b, math::BinomialCoefficient(n, a) * 
				math::BinomialCoefficient(a, b) / 
				(two_raised_to_a * n_factorial));
	  if(n % 2 == 1) {
	    n_th_order_matrix.set(a, b, -n_th_order_matrix.get(a, b));
	  }
	}

	two_raised_to_a *= 2.0;
      }

      n_factorial *= (n + 1);
    }
  }

  int get_dimension() const { return dim_; }

  int get_total_num_coeffs(int order) const;

  int get_max_total_num_coeffs() const;

  int get_max_order() const;

  /** @brief Computes the position of the given multiindex.
   */
  int ComputeMultiindexPosition(const ArrayList<int> &multiindex) const;

  /** @brief Computes the computational cost of evaluating a far-field
   *         expansion of order p at a single query point.
   */
  double FarFieldEvaluationCost(int order) const;

  /** @brief Computes the compuational cost of translating a far-field
   *         moment of order p into a local moment of the same order.
   */
  double FarFieldToLocalTranslationCost(int order) const;

  /** @brief Computes the computational cost of directly accumulating
   *         a single reference point into a local moment of order p.
   */
  double DirectLocalAccumulationCost(int order) const;

  /** @brief Initialize the auxiliary object with precomputed
   *         quantities for order up to max_order for the given
   *         dimensionality.
   */
  void Init(int max_order, int dim);

  /** @brief Print useful information about this object.
   */
  void PrintDebug(const char *name="", FILE *stream=stderr) const;

};

#endif
