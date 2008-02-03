/*
 * =====================================================================================
 * 
 *       Filename:  non_convex_mvu_impl.h
 * 
 *    Description:  
 * 
 *        Version:  1.0
 *        Created:  02/03/2008 11:45:39 AM EST
 *       Revision:  none
 *       Compiler:  gcc
 * 
 *         Author:  Nikolaos Vasiloglou (NV), nvasil@ieee.org
 *        Company:  Georgia Tech Fastlab-ESP Lab
 * 
 * =====================================================================================
 */

// Give the coordinates and the gradient in the transpose format
// dimension x num_of_points
void NonConvexMVU::ComputeGradient_(Matrix &cooridnates, Matrix &gradient) {
  index_t num_of_constraints = neighbors_.size();
  index_t new_dimension = coordinate.n_rows();
  gradient.Copy(coordinates);
  for(index_t i=0; i<gradient.n_cols(); i++) {
    for(index_t k=0; k<knns_; k++) {
      double a_i_r[new_dimension];
      double *point1 = coordinate.GetColumnPtr(i);
      double *point2 = coordinate.GetColumnPtr(neighbors_[i*knns_+k]);
      la::AddOvewrite(p1, p2, a_i_r);
      double dist_diff = (la::DistanceSqEuclidean(new_dimension, p1, p2) 
                          -distances[i*knns_+k]) *sigma_;
      la::AddExpert(new_dimension,
          2 * (-langrange_mult[j]+dist_diff), 
          a_i_r, 
          gradient.GetColumnPtr(i));
      memcpy(gradient.GetColumnPtr(neighbors_[i*knns_+k]),
             gradient.GetColumnPtr(i),
             new_dimension*sizeof(double));

    }
  }   
}

NonConvexMVU::Compute(index_t new_dimension, Matrix *new_coordinates) {
  new_coordinates->Init(num_of_points_, new_dimension);
  
}
