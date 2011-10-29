#include "gen_cone_tree_impl.h"


namespace tree_gen_cone_tree_private {
  // fixed!!  
  size_t FurthestColumnIndex(const arma::vec& pivot,
			     const arma::mat& matrix, 
			     size_t begin, size_t count,
			     double *furthest_cosine) {
    
    size_t furthest_index = -1;
    size_t end = begin + count;
    *furthest_cosine = 1.0;

    for(size_t i = begin; i < end; i++) {
      double cosine_between_center_and_point = 
	Cosine::Evaluate(pivot, matrix.unsafe_col(i));
      
      if((*furthest_cosine) > cosine_between_center_and_point) {
	*furthest_cosine = cosine_between_center_and_point;
	furthest_index = i;
      }
    }

    assert((*furthest_cosine) >= -1.0);

    return furthest_index;
  }

};
