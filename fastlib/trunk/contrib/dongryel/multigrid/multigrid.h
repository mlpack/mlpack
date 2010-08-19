/** @file multigrid.h
 *  @brief A prototype of multigrid algorithm for solving linear systems.
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#ifndef CONTRIB_DONGRYEL_MULTIGRID_MULTIGRID_H
#define CONTRIB_DONGRYEL_MULTIGRID_MULTIGRID_H

#include <vector>
#include "fastlib/fastlib.h"
#include "multigrid_level.h"

namespace fl {
namespace ml {
template<typename MatrixType, typename VectorType>
class Multigrid {
  private:

    int level_threshold_;

    /** @brief The left hand side, A matrix in Ax = b.
    */
    MatrixType *left_hand_side_;

    /** @brief The right hand side, b vector in Ax = b.
     */
    VectorType *right_hand_side_;

    /** @brief The maximum number of iterations.
     */
    int max_num_iterations_;

    /** @brief The list of coarsened problems. Higher indices
     *         imply coarser problems.
     */
    std::vector<MultigridLevel *> levels_;

  private:

    void Coarsen_(
      const MultigridLevel &level_in,
      MultigridLevel *coarsened_level_out);

  public:

    Multigrid();

    ~Multigrid();

    void Init(
      MatrixType &left_hand_side_in,
      VectorType &right_hand_side_in,
      int level_threshold_in,
      int max_num_iterations_in);

    void Compute(Vector *output);
};
};
};

#endif
