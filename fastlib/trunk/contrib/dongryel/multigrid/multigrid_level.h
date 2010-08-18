/** @file multigrid_level.h
 *  @brief A multigrid level generated during the coarsening procedure.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef MLPACK_MULTIGRID_MULTIGRID_LEVEL_H
#define MLPACK_MULTIGRID_MULTIGRID_LEVEL_H

#include <vector>
#include "fastlib/fastlib.h"

namespace fl {
namespace ml {
class MultigridLevel {
  private:

    /** @brief The indices of the points chosen by the coarsening
     *         procedure.
     */
    std::vector<int> point_indices_;

    /** @brief The coarser left hand side created by the
     *         coarsening procedure.
     */
    Matrix left_hand_side_;

    /** @brief The coarser right hand sides created by the
     *         coarsening procedure.
     */
    Vector right_hand_side_;

    /** @brief The interpolation weights.
     */
    Matrix interpolation_weights_;

  public:

    void Build(
      const MultigridLevel &previous_level_in,
      const std::vector< std::pair<int, int> >
      &coarse_physical_index_and_label_pairs) {

      // First, set the coarse point indices to be the indices on the
      // current level.
      point_indices_.resize(coarse_physical_index_and_label_pairs.size());
      for (int i = 0; i < point_indices_.size(); i++) {
        point_indices_[i] = coarse_physical_index_and_label_pairs[i].second;
      }

      // Next, build the interpolation matrix.
      const std::vector<int> &previous_level_point_indices =
        previous_level_in.point_indices();
      interpolation_weights_.Init(
        previous_level_point_indices.size(), point_indices_.size());

      for (unsigned int i = 0; i < previous_level_point_indices.size(); i++) {
        double affinity_row_sum = 0;
        for (unsigned int j = 0; j < point_indices_.size(); j++) {
          affinity_row_sum += previous_level_in.get(
                                i, coarse_physical_index_and_label_pairs[j].first);
        }
        for (unsigned int j = 0; j < point_indices_.size(); j++) {
          int coarse_point_physical_index =
            coarse_physical_index_and_label_pairs[j].first;
          interpolation_weights_.set(
            i, coarse_point_physical_index,
            previous_level_in.get(i, coarse_point_physical_index) /
            affinity_row_sum);
        }
      }

      // Using the interpolation matrix, build the coarsened left hand side.
      const Matrix &previous_level_left_hand_side =
        previous_level.left_hand_side();
      left_hand_side_.Init(point_indices_.size(), point_indices_.size());
      for (unsigned int l = 0; l < point_indices_.size(); l++) {
        for (unsigned int k = 0; k < point_indices_.size(); k++) {
          double new_accumulant = 0;
          for (unsigned int j = 0; j < interpolation_weights_.n_cols(); j++) {
            for (unsigned int i = 0; i < interpolation_weights_.n_rows(); i++) {
              new_accumulant += interpolation_weights_.get(i, k) *
                                previous_level_left_hand_side.get(i, j) *
                                interpolation_weights_.get(j, l);
            }
          }
          new_accumulant = 0.5 * new_accumulant;
          left_hand_side_.set(k, l, new_acumulant);
        }
      }

      // Using the interpolation matrix, build the coarsened right
      // hand side.
      right_hand_side_.Init(point_indices_.size());
      const Vector &previous_level_right_hand_side =
        previous_level.right_hand_side();
      for (unsigned int i = 0; i < point_indices_.size(); i++) {
        double dot_product = 0;
        for (unsigned int j = 0;
             j < previous_level_right_hand_side.size(); j++) {
          dot_product += interpolation_matrix_.get(j, i) *
                         previous_level_right_hand_side[j];
        }
        right_hand_side_[i] = dot_product;
      }
    }

    double get(int row, int col) const {
      return left_hand_side_.get(row, col);
    }

    const std::vector<int> &point_indices() const {
      return point_indices_;
    }

    std::vector<int> &point_indices() {
      return point_indices_;
    }

    const Matrix &left_hand_side() const {
      return left_hand_side_;
    }

    Matrix &left_hand_side() {
      return left_hand_side_;
    }

    const Vector &right_hand_side() const {
      return right_hand_side_;
    }

    Vector &right_hand_side() {
      return right_hand_side_;
    }

    const Matrix &interpolation_weights() const {
      return interpolation_weights_;
    }

    Matrix &interpolation_weights() {
      return interpolation_weights_;
    }
};
};
};

#endif
