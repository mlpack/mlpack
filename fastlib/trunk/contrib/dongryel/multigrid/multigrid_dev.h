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

  const double threshold = 0.2;

  // Make a copy of the fine nodes from the points owned by the
  // previous level.
  const std::vector<int> &fine_point_indices = level_in.point_indices();
  std::vector<int> shuffle_indices(fine_point_indices.size());
  for (int i = 0; i < shuffle_indices.size(); i++) {
    shuffle_indices[i] = i;
  }
  std::random_shuffle(shuffle_indices.begin(), shuffle_indices.end());

  // The generated coarse points.
  std::vector< std::pair<int, int> > coarse_point_indices;

  for (int i = 0; i < fine_point_indices.size(); i++) {

    // The index of the fine node point.
    int fine_point_index = shuffle_indices[i];

    // The associated label of the fine node point.
    int fine_point_label = fine_point_indices[ fine_point_index ];

    // Compute the sum of the affinities between the current fine node
    // point and the existing set of coarse points.
    double sum_coarse_affinities = 0;
    for (int j = 0; j < coarse_point_indices.size(); j++) {

      // The index of the coarse node point.
      int coarse_point_index = coarse_point_indices[j].first;
      sum_coarse_affinities +=
        level_in.get(fine_point_index, coarse_point_index);
    }

    // Compute the sum of the affinities between the current fine node
    // and all of the points.
    double sum_all_affinities = 0;
    for (int j = 0; j < fine_point_indices.size(); j++) {
      sum_all_affinities += level_in.get(fine_point_index, j);
    }

    // Add to the coarse set if the following condition is satisfied.
    if (sum_affinities < threshold * sum_all_affinities) {
      coarse_point_indices.push_back(
        std::pair<int, int>(fine_point_index, fine_point_label));
    }
  } // end of looping over all fine nodes.

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
