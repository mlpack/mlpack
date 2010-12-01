/** @author Dongryeol Lee
 *
 *  @brief The generic L-BFGS optimizer.
 *
 *  @file lbfgs.h
 */

#ifndef CORE_OPTIMIZATION_LBFGS_H
#define CORE_OPTIMIZATION_LBFGS_H

#include "core/table/dense_matrix.h"
#include "core/table/dense_point.h"

namespace core {
namespace optimization {
template<typename FunctionType>
class Lbfgs {

  public:

    /** @brief The set of parameters for the L-BFGS routine.
     */
    class LbfgsParam {
      private:

        /** @brief Parameter to control the accuracy of the line search
         *         routine for determining the Armijo condition.
         */
        double armijo_constant_;

        /** @brief The minimum step of the line search routine.
         */
        double min_step_;

        /** @brief The maximum step of the line search routine.
         */
        double max_step_;

        /** @brief The maximum number of trials for the line search.
         */
        int max_line_search_;

        /** @brief Parameter for detecting Wolfe condition.
         */
        double wolfe_;

      public:

        void set_max_num_line_searches(int max_num_line_searches_in);

        double armijo_constant() const;

        double min_step() const;

        double max_step() const;

        int max_line_search() const;

        double wolfe() const;

        LbfgsParam();
    };

  private:

    LbfgsParam param_;

    FunctionType *function_;

    core::table::DensePoint new_iterate_tmp_;

    core::table::DenseMatrix s_lbfgs_;

    core::table::DenseMatrix y_lbfgs_;

    int num_basis_;

    std::pair< core::table::DensePoint, double > min_point_iterate_;

  private:

    double Evaluate_(const core::table::DensePoint &iterate);

    double ChooseScalingFactor_(
      int iteration_num,
      const core::table::DensePoint &gradient);

    bool GradientNormTooSmall_(
      const core::table::DensePoint &gradient);

    bool LineSearch_(
      double &function_value,
      core::table::DensePoint &iterate,
      core::table::DensePoint &gradient,
      const core::table::DensePoint &search_direction,
      double &step_size);

    void SearchDirection_(
      const core::table::DensePoint &gradient,
      int iteration_num, double scaling_factor,
      core::table::DensePoint *search_direction);

    void UpdateBasisSet_(
      int iteration_num,
      const core::table::DensePoint &iterate,
      const core::table::DensePoint &old_iterate,
      const core::table::DensePoint &gradient,
      const core::table::DensePoint &old_gradient);

  public:

    const std::pair< core::table::DensePoint, double > &min_point_iterate() const;

    void Init(FunctionType &function_in, int num_basis);

    void set_max_num_line_searches(int max_num_line_searches_in);

    bool Optimize(
      int num_iterations,
      core::table::DensePoint *iterate);
};
};
};

#endif
