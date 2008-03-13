/*
 * =====================================================================================
 * 
 *       Filename:  optimization_utils.h
 * 
 *    Description
 * 
 *        Version:  1.0
 *        Created:  03/12/2008 06:29:51 PM EDT
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */
#include "fastlib/fastlib.h"

class OptUtils {
 public:
  static void RemoveMean(Matrix *data) {
    index_t dimension=data->n_rows();
    index_t num_of_points=data->n_cols();
    Vector mean;
    mean.Init(dimension);
    mean.SetAll(0);
    for(index_t i=0; i<num_of_points; i++){
      la::AddTo(dimension, data->GetColumnPtr(i), mean.ptr());
    }
    la::Scale(-1.0/num_of_points, &mean);
    for(index_t i=0; i<num_of_points; i++){
      la::AddTo(dimension, mean.ptr(), data->GetColumnPtr(i));
    }
  }
  static void SparseProjection(Matrix *data, double sparse_factor);
};
