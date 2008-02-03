/*
 * =====================================================================================
 * 
 *       Filename:  non_convex_mvu.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/03/2008 11:33:23 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

#ifndef NON_CONVEX_MVU_H__
#define NON_CONVEX_MVU_H__

#include <string>
#include "fastlib/fastlib.h"
#include "mlpack/allknn/all_knn.h"

class NonConvexMVU {
 public:
  void Init(std::string filename);
  void ComputeNeigborhoods(index_t knns);
  void Compute(index_t new_dimension, Matrix *new_coordinates);
 private:
  AllkNN allknn_;
  index_t num_of_points_;
  index_t dimension_;
  index_t knns_;
  Vector distances_;
  ArrayList<indext_> neighbors_;
  Vector lagrange_mult_;
  double sigma_;
  void ComputeGradient_(Matrix &cooridnate, Matrix &gradient);
};

#include "non_convex_mvu_impl.h"
#endif //NON_CONVEX_MVU_H__
