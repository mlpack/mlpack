#include "gen_metric_tree_impl.h"


namespace tree_gen_metric_tree_private {

  size_t FurthestColumnIndex(const arma::vec& pivot, const arma::mat& matrix, 
			     size_t begin, size_t count,
			     double *furthest_distance) {
    
    size_t furthest_index = -1;
    size_t end = begin + count;
    *furthest_distance = -1.0;

    for(size_t i = begin; i < end; i++) {
      double distance_between_center_and_point = 
	mlpack::kernel::SquaredEuclideanDistance::Evaluate(pivot, matrix.unsafe_col(i));
      
      if((*furthest_distance) < distance_between_center_and_point) {
	*furthest_distance = distance_between_center_and_point;
	furthest_index = i;
      }
    }

    return furthest_index;
  }

};
