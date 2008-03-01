#ifndef MULTIINDEX_UTIL_H
#define MULTIINDEX_UTIL_H

#include "fastlib/fastlib.h"

class MultiIndexUtil {
  
  public:
    /** @brief Computes the polynomial expansion in terms of
     *         multiindex for a D-dimensional point.
     *
     *  @param dimension The dimensionality.
     *  @param point The pointer array of doubles containing the coordinates.
     *  @param point_expansion The computed multiindex expansion of the
     *                         point.
     */
    static void ComputePointMultivariatePolynomial(int dimension, 
						   int lpr_order,
						   const double *point,
						   double *point_expansion) {

      // Temporary variables for multiindex looping
      ArrayList<int> heads;
      heads.Init(dimension + 1);
      for(index_t i = 0; i < dimension; i++) {
        heads[i] = 0;
      }
      heads[dimension] = INT_MAX;
      
      point_expansion[0] = 1.0;
      for(index_t k = 1, t = 1, tail = 1; k <= lpr_order; k++, tail = t) {
	for(index_t i = 0; i < dimension; i++) {
	  int head = (int) heads[i];
	  heads[i] = t;
	  for(index_t j = head; j < tail; j++, t++) {	  
	    point_expansion[t] = point_expansion[j] * point[i];
	  }
	}
      }
    }

    static void ComputePointMultivariatePolynomial(int dimension, 
						   int lpr_order,
						   const Matrix &points,
						   Matrix &point_expansions) {
      
      for(index_t i = 0; i < points.n_cols(); i++) {
	ComputePointMultivariatePolynomial(dimension, lpr_order,
					   points.GetColumnPtr(i),
					   point_expansions.GetColumnPtr(i));
      }
    }
};

#endif
