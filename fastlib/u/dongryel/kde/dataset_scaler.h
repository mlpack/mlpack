/**
 * @file dataset_scaler.h
 *
 * This file contains utility functions to scale the given query and
 * reference dataset pair.
 *
 * @author Dongryeol Lee (dongryel)
 * @bug No known bugs.
 */

#ifndef DATASET_SCALER_H
#define DATASET_SACLER_H

#include <fastlib/fastlib.h>

class DatasetScaler {

 public:

  static void ScaleDataByMinMax(Matrix &qset, Matrix &rset,
				bool queries_equal_references) {
    
    int num_dims = rset.n_rows();
    DHrectBound<2> qset_bound;
    DHrectBound<2> rset_bound;
    qset_bound.Init(qset.n_rows());
    rset_bound.Init(qset.n_rows());

    // go through each query/reference point to find out the bounds
    for(index_t r = 0; r < rset.n_cols(); r++) {
      Vector ref_vector;
      rset.MakeColumnVector(r, &ref_vector);
      rset_bound |= ref_vector;
    }
    for(index_t q = 0; q < qset.n_cols(); q++) {
      Vector query_vector;
      qset.MakeColumnVector(q, &query_vector);
      qset_bound |= query_vector;
    }

    for(index_t i = 0; i < num_dims; i++) {
      DRange qset_range = qset_bound.get(i);
      DRange rset_range = rset_bound.get(i);
      double min_coord = min(qset_range.lo, rset_range.lo);
      double max_coord = max(qset_range.hi, rset_range.hi);
      double width = max_coord - min_coord;

      printf("Dimension %d range: [%g, %g]\n", i, min_coord, max_coord);

      for(index_t j = 0; j < rset.n_cols(); j++) {
	rset.set(i, j, (rset.get(i, j) - min_coord) / width);
      }

      if(!queries_equal_references) {
	for(index_t j = 0; j < qset.n_cols(); j++) {
	  qset.set(i, j, (qset.get(i, j) - min_coord) / width);
	}
      }
    }
  }

};

#endif
