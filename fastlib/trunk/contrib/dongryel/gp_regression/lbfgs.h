/** @author Dongryeol Lee
 *
 *  @brief The generic L-BFGS optimizer.
 *
 *  @file lbfgs.h
 */

#ifndef OPTIMIZATION_LBFGS_H
#define OPTIMIZATION_LBFGS_H

#include "fastlib/la/matrix.h"

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

    Vector new_iterate_tmp_;

    Matrix s_lbfgs_;

    Matrix y_lbfgs_;

    int num_basis_;

    std::pair< Vector, double > min_point_iterate_;

  private:

    double Evaluate_(const Vector &iterate);

    double ChooseScalingFactor_(
      int iteration_num,
      const Vector &gradient);

    bool GradientNormTooSmall_(
      const Vector &gradient);

    bool LineSearch_(double &function_value,
                     Vector &iterate,
                     Vector &gradient,
                     const Vector &search_direction,
                     double &step_size);

    void SearchDirection_(const Vector &gradient,
                          int iteration_num, double scaling_factor,
                          Vector *search_direction);

    void UpdateBasisSet_(
      int iteration_num,
      const Vector &iterate,
      const Vector &old_iterate,
      const Vector &gradient,
      const Vector &old_gradient);

  public:

    const std::pair< Vector, double > &min_point_iterate() const;

    void Init(FunctionType &function_in, int num_basis);

    void set_max_num_line_searches(int max_num_line_searches_in);

    bool Optimize(int num_iterations,
                  Vector *iterate);
};
};

#endif
