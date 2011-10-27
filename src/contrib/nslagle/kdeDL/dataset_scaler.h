/** @file dataset_scaler.h
 *
 *  This file contains utility functions to scale the given query and
 *  reference dataset pair.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @bug No known bugs.
 */

#ifndef DATASET_SCALER_H
#define DATASET_SCALER_H

#include <mlpack/core.h>
#include <mlpack/core/tree/bounds.hpp>

using namespace mlpack::math;

/** @brief A static class providing utilities for scaling the query
 *         and the reference datasets.
 *
 *  Example use:
 *
 *  @code
 *    DatasetScaler::ScaleDataByMinMax(qset, rset, queries_equal_references);
 *  @endcode
 */
class DatasetScaler {

 public:

  /** @brief Scale the given query and the reference datasets to lie
   *         in the positive quadrant.
   *
   *  @param qset The column-oriented query set.
   *  @param rset The column-oriented reference set.
   *  @param queries_equal_references The boolean flag that tells whether
   *                                  the queries equal the references.
   */
  static void TranslateDataByMin(arma::mat &qset, arma::mat &rset,
				 bool queries_equal_references) {
    
    size_t num_dims = rset.n_rows;
    bound::HRectBound<2> qset_bound = bound::HRectBound<2> (qset.n_rows);;
    bound::HRectBound<2> rset_bound = bound::HRectBound<2> (qset.n_rows);;

    // go through each query/reference point to find out the bounds
    for(size_t r = 0; r < rset.n_cols; r++) {
      arma::vec ref_vector = rset.unsafe_col(r);
      rset_bound |= ref_vector;
    }
    for(size_t q = 0; q < qset.n_cols; q++) {
      arma::vec query_vector = qset.unsafe_col(q);
      qset_bound |= query_vector;
    }

    for(size_t i = 0; i < num_dims; i++) {
      Range qset_range = qset_bound[i];
      Range rset_range = rset_bound[i];
      double min_coord = std::min(qset_range.lo, rset_range.lo);
      double max_coord = std::max(qset_range.hi, rset_range.hi);

      printf("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);

      for(size_t j = 0; j < rset.n_cols; j++) {
	rset(i, j) = rset(i, j) - min_coord;
      }

      if(!queries_equal_references) {
	for(size_t j = 0; j < qset.n_cols; j++) {
	  qset(i, j) = qset(i, j) - min_coord;
	}
      }
    }
  }

  /** @brief Scale the given query and the reference datasets to fit
   *         in the unit hypercube $[0,1]^D$ where $D$ is the common
   *         dimensionality of the two datasets. 
   *
   *  @param qset The column-oriented query set.
   *  @param rset The column-oriented reference set.
   *  @param queries_equal_references The boolean flag that tells whether
   *                                  the queries equal the references.
   */
  static void ScaleDataByMinMax(arma::mat &qset, arma::mat &rset,
				bool queries_equal_references) {
    
    size_t num_dims = qset.n_rows;
    bound::HRectBound<2> total_bound = bound::HRectBound<2> (qset.n_rows);

    // go through each query/reference point to find out the bounds
    for(size_t r = 0; r < rset.n_cols; r++) {
      arma::vec ref_vector = rset.unsafe_col(r);
      total_bound |= ref_vector;
    }
    if(!queries_equal_references) {
      for(size_t q = 0; q < qset.n_cols; q++) {
        arma::vec query_vector = qset.unsafe_col(q);
	total_bound |= query_vector;
      }
    }

    for(size_t i = 0; i < num_dims; i++) {
      Range total_range = total_bound[i];
      double min_coord = total_range.lo;
      double max_coord = total_range.hi;
      double width = max_coord - min_coord;

      printf("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);

      for(size_t j = 0; j < rset.n_cols; j++) {
	if(width > 0) {
	  rset(i, j) = (rset(i, j) - min_coord) / width;
	}
	else {
	  rset(i, j) = 0;
	}
      }

      if(!queries_equal_references) {
	for(size_t j = 0; j < qset.n_cols; j++) {
	  if(width > 0) {
	    qset(i, j) = (qset(i, j) - min_coord) / width;
	  }
	  else {
	    qset(i, j) = 0;
	  }
	}
      }
    }
  }

  /** @brief Standardize the given query and the reference datasets in
   *         each dimension to have zero mean and at most unit
   *         variance.
   *
   *         Assumes that the query and the reference together contain
   *         more than one instance.
   *
   *  @param qset The column-oriented query set.
   *  @param rset The column-oriented reference set.
   *  @param queries_equal_references The boolean flag that tells whether
   *                                  the queries equal the references.
   */
  static void StandardizeData(arma::mat &qset, arma::mat &rset,
			      bool queries_equal_references) {

    arma::vec mean_vector(qset.n_rows);
    arma::vec standard_deviation_vector(qset.n_rows);

    mean_vector.zeros();
    standard_deviation_vector.zeros();

    // Go through each query/reference point to find out the mean
    // vectors.
    for(size_t r = 0; r < rset.n_cols; r++) {
      mean_vector += rset.unsafe_col(r);
    }
    if(!queries_equal_references) {
      for(size_t q = 0; q < qset.n_cols; q++) {
        mean_vector += qset.unsafe_col(q);
      }
      mean_vector *= 1.0 / ((double) qset.n_cols + rset.n_cols);
    }
    else {
      mean_vector *= 1.0 / ((double) qset.n_cols);
    }

    // Now find out the standard deviation along each dimension.
    for(size_t r = 0; r < rset.n_cols; r++) {
      for(size_t i = 0; i < rset.n_rows; i++) {
	standard_deviation_vector[i] += (rset(i, r) - mean_vector[i]) * (rset(i, r) - mean_vector[i]);
      }
    }
    if(!queries_equal_references) {
      for(size_t q = 0; q < qset.n_cols; q++) {
	for(size_t i = 0; i < qset.n_rows; i++) {
	  standard_deviation_vector[i] += (qset(i, q) - mean_vector[i]) * (qset(i, q) - mean_vector[i]);
	}
      }
      standard_deviation_vector *= 1.0 / ((double) qset.n_cols + rset.n_cols - 1);
    }
    else {
      standard_deviation_vector *= 1.0 / ((double) rset.n_cols);
    }

    // Now scale the datasets using the computed mean and the standard
    // deviation.
    for(size_t r = 0; r < rset.n_cols; r++) {
      for(size_t d = 0; d < rset.n_rows; d++) {
	rset(d, r) = (rset(d, r) - mean_vector[d]) / 
		 standard_deviation_vector[d];
      }
    }
    if(!queries_equal_references) {
      for(size_t q = 0; q < qset.n_cols; q++) {
	for(size_t d = 0; d < qset.n_rows; d++) {
	  qset(d, q) = (qset(d, q) - mean_vector[d]) /
		   standard_deviation_vector[d];
	}
      }
    }
  }

};

#endif
