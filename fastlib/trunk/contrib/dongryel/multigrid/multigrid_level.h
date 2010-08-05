/** @file multigrid_level.h
 *  @brief A multigrid level generated during the coarsening procedure.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CONTRIB_DONGRYEL_MULTIGRID_MULTIGRID_LEVEL_H
#define CONTRIB_DONGRYEL_MULTIGRID_MULTIGRID_LEVEL_H

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
    SparseMatrix left_hand_side_;

    /** @brief The coarser right hand sides created by the
     *         coarsening procedure.
     */
    Vector right_hand_side_;

    /** @brief The interpolation weights.
     */
    Matrix interpolation_weights_;

  public:

    const std::vector<int> &point_indices() const {
      return point_indices_;
    }

    std::vector<int> &point_indices() {
      return point_indices_;
    }

    const SparseMatrix &left_hand_side() const {
      return left_hand_side_;
    }

    SparseMatrix &left_hand_side() {
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
