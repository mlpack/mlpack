/** @file multigrid_dev.h
 *  @brief An implementation of multigrid algorithm for solving linear systems.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CONTRIB_DONGRYEL_MULTIGRID_MULTIGRID_DEV_H
#define CONTRIB_DONGRYEL_MULTIGRID_MULTIGRID_DEV_H

#include <algorithm>
#include "multigrid.h"

namespace fl {
namespace ml {

template<typename MatrixType, typename VectorType>
void Multigrid<MatrixType, VectorType>::Coarsen_(
  const MultigridLevel &level_in,
  MultigridLevel *coarsened_level_out) {

  // Make a copy of the fine nodes from the points owned by the
  // previous level.
  std::vector<int> fine_point_indices = level_in.point_indices();
  std::random_shuffle(fine_point_indices.begin(), fine_point_indices.end());

  for (int i = 0; i < fine_point_indices.size(); i++) {

  }
}

template<typename MatrixType, typename VectorType>
void Multigrid<MatrixType, VectorType>::Init(
  MatrixType &left_hand_side_in,
  VectorType &right_hand_side_in,
  int max_num_iterations_in) {

  // Set the incoming variables.
  left_hand_side_ = &left_hand_side_in;
  right_hand_side_ = &right_hand_side_in;
  max_num_iterations_ = max_num_iterations_in;

  // Generate the coarse problems.

}

template<typename MatrixType, typename VectorType>
void Multigrid<MatrixType, VectorType>::Compute(Vector *output) {

  // Allocate space for the output vector.
  output->Init(right_hand_side_->length());
  output->SetZero();


}
};
};

#endif
