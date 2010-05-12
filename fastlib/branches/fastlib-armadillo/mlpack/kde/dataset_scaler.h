/** @file dataset_scaler.h
 *
 *  This file contains utility functions to scale the given query and
 *  reference dataset pair.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *  @bug No known bugs.
 */

#ifndef DATASET_SCALER_H
#define DATASET_SACLER_H

#include <fastlib/fastlib.h>

#include <armadillo>
#include <fastlib/base/arma_compat.h>

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
  static void TranslateDataByMin(Matrix &qset, Matrix &rset,
				 bool queries_equal_references) {
    
    int num_dims = rset.n_rows();
    DHrectBound<2> qset_bound(qset.n_rows());
    DHrectBound<2> rset_bound(qset.n_rows());

    // go through each query/reference point to find out the bounds
    for(index_t r = 0; r < rset.n_cols(); r++) {
      Vector ref_vector;
      rset.MakeColumnVector(r, &ref_vector);
      arma::vec tmp;
      arma_compat::vectorToVec(ref_vector, tmp);
      rset_bound |= tmp;
    }
    for(index_t q = 0; q < qset.n_cols(); q++) {
      Vector query_vector;
      qset.MakeColumnVector(q, &query_vector);
      arma::vec tmp;
      arma_compat::vectorToVec(query_vector, tmp);
      qset_bound |= tmp;
    }

    for(index_t i = 0; i < num_dims; i++) {
      DRange qset_range = qset_bound[i];
      DRange rset_range = rset_bound[i];
      double min_coord = min(qset_range.lo, rset_range.lo);
      double max_coord = max(qset_range.hi, rset_range.hi);

      printf("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);

      for(index_t j = 0; j < rset.n_cols(); j++) {
	rset.set(i, j, rset.get(i, j) - min_coord);
      }

      if(!queries_equal_references) {
	for(index_t j = 0; j < qset.n_cols(); j++) {
	  qset.set(i, j, qset.get(i, j) - min_coord);
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
  static void ScaleDataByMinMax(Matrix &qset, Matrix &rset,
				bool queries_equal_references) {
    
    index_t num_dims = qset.n_rows();
    DHrectBound<2> total_bound(qset.n_rows());

    // go through each query/reference point to find out the bounds
    for(index_t r = 0; r < rset.n_cols(); r++) {
      Vector ref_vector;
      rset.MakeColumnVector(r, &ref_vector);
      arma::vec tmp;
      arma_compat::vectorToVec(ref_vector, tmp);
      total_bound |= tmp;
    }
    if(!queries_equal_references) {
      for(index_t q = 0; q < qset.n_cols(); q++) {
	Vector query_vector;
	qset.MakeColumnVector(q, &query_vector);
        arma::vec tmp;
        arma_compat::vectorToVec(query_vector, tmp);
	total_bound |= tmp;
      }
    }

    for(index_t i = 0; i < num_dims; i++) {
      DRange total_range = total_bound[i];
      double min_coord = total_range.lo;
      double max_coord = total_range.hi;
      double width = max_coord - min_coord;

      printf("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);

      for(index_t j = 0; j < rset.n_cols(); j++) {
	if(width > 0) {
	  rset.set(i, j, (rset.get(i, j) - min_coord) / width);
	}
	else {
	  rset.set(i, j, 0);
	}
      }

      if(!queries_equal_references) {
	for(index_t j = 0; j < qset.n_cols(); j++) {
	  if(width > 0) {
	    qset.set(i, j, (qset.get(i, j) - min_coord) / width);
	  }
	  else {
	    qset.set(i, j, 0);
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
  static void StandardizeData(Matrix &qset, Matrix &rset,
			      bool queries_equal_references) {

    Vector mean_vector, standard_deviation_vector;

    mean_vector.Init(qset.n_rows());
    mean_vector.SetZero();
    standard_deviation_vector.Init(qset.n_rows());
    standard_deviation_vector.SetZero();

    // Go through each query/reference point to find out the mean
    // vectors.
    for(index_t r = 0; r < rset.n_cols(); r++) {
      la::AddTo(rset.n_rows(), rset.GetColumnPtr(r), mean_vector.ptr());
    }
    if(!queries_equal_references) {
      for(index_t q = 0; q < qset.n_cols(); q++) {
	la::AddTo(qset.n_rows(), qset.GetColumnPtr(q), mean_vector.ptr());
      }
      la::Scale(qset.n_rows(), 1.0 / ((double) qset.n_cols() + rset.n_cols()),
		mean_vector.ptr());
    }
    else {
      la::Scale(qset.n_rows(), 1.0 / ((double) qset.n_cols()),
		mean_vector.ptr());
    }

    // Now find out the standard deviation along each dimension.
    for(index_t r = 0; r < rset.n_cols(); r++) {
      for(index_t i = 0; i < rset.n_rows(); i++) {
	standard_deviation_vector[i] += 
	  math::Sqr(rset.get(i, r) - mean_vector[i]);
      }
    }
    if(!queries_equal_references) {
      for(index_t q = 0; q < qset.n_cols(); q++) {
	for(index_t i = 0; i < qset.n_rows(); i++) {
	  standard_deviation_vector[i] +=
	    math::Sqr(qset.get(i, q) - mean_vector[i]);
	}
      }
      la::Scale(qset.n_rows(), 
		1.0 / ((double) qset.n_cols() + rset.n_cols() - 1),
		standard_deviation_vector.ptr());
    }
    else {
      la::Scale(rset.n_rows(), 1.0 / ((double) rset.n_cols()),
		standard_deviation_vector.ptr());
    }

    // Now scale the datasets using the computed mean and the standard
    // deviation.
    for(index_t r = 0; r < rset.n_cols(); r++) {
      for(index_t d = 0; d < rset.n_rows(); d++) {
	rset.set(d, r, (rset.get(d, r) - mean_vector[d]) / 
		 standard_deviation_vector[d]);
      }
    }
    if(!queries_equal_references) {
      for(index_t q = 0; q < qset.n_cols(); q++) {
	for(index_t d = 0; d < qset.n_rows(); d++) {
	  qset.set(d, q, (qset.get(d, q) - mean_vector[d]) /
		   standard_deviation_vector[d]);
	}
      }
    }
  }

};

#endif
