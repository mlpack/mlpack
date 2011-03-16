/** @author Dongryeol Lee
 *
 *  @brief The generic L-BFGS optimizer.
 *
 *  @file lbfgs.h
 */

#ifndef CORE_OPTIMIZATION_LBFGS_H
#define CORE_OPTIMIZATION_LBFGS_H

#include <armadillo>

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

    arma::vec new_iterate_tmp_;

    arma::mat s_lbfgs_;

    arma::mat y_lbfgs_;

    int num_basis_;

    std::pair< arma::vec, double > min_point_iterate_;

  private:

    double Evaluate_(const arma::vec &iterate);

    double ChooseScalingFactor_(
      int iteration_num,
      const arma::vec &gradient);

    bool GradientNormTooSmall_(
      const arma::vec &gradient);

    bool LineSearch_(
      double &function_value,
      arma::vec &iterate,
      arma::vec &gradient,
      const arma::vec &search_direction,
      double &step_size);

    void SearchDirection_(
      const arma::vec &gradient,
      int iteration_num, double scaling_factor,
      arma::vec *search_direction);

    void UpdateBasisSet_(
      int iteration_num,
      const arma::vec &iterate,
      const arma::vec &old_iterate,
      const arma::vec &gradient,
      const arma::vec &old_gradient);

  public:

    const std::pair< arma::vec, double > &min_point_iterate() const;

    void Init(FunctionType &function_in, int num_basis);

    void set_max_num_line_searches(int max_num_line_searches_in);

    bool Optimize(
      int num_iterations,
      arma::vec *iterate);
};
};
};

#endif
