/** @author Dongryeol Lee
 *
 *  @brief The generic L-BFGS optimizer.
 *
 *  @file lbfgs.h
 */

#ifndef OPTIMIZATION_LBFGS_H
#define OPTIMIZATION_LBFGS_H

#include <fastlib/fastlib.h>
#include <armadillo>

namespace mlpack {
namespace optimization {

template<typename FunctionType>
class L_BFGS {

  // These parameters should be replaced with the IO system when mamidon is
  // finished with it (#65).
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

    FunctionType function_;

    arma::mat new_iterate_tmp_;
    arma::cube s_lbfgs_; // stores all the s matrices in memory
    arma::cube y_lbfgs_; // stores all the y matrices in memory

    int num_basis_;

    std::pair<arma::mat, double> min_point_iterate_;

  private:
    /***
     * Evaluate the function at the given iterate point and store the result if
     * it is a new minimum.
     *
     * @return The value of the function
     */
    double Evaluate_(const arma::mat& iterate);

    /***
     * Calculate the scaling factor gamma which is used to scale the Hessian
     * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal
     * (1989).
     *
     * @return The calculated scaling factor
     */
    double ChooseScalingFactor_(int iteration_num, const arma::mat& gradient);

    /***
     * Check to make sure that the norm of the gradient is not smaller than
     * 1e-5.  Currently that value is not configurable.
     *
     * @return (norm < 1e-5)
     */
    bool GradientNormTooSmall_(const arma::mat& gradient);

    /***
     * Perform a back-tracking line search along the search direction to
     * calculate a step size satisfying the Wolfe conditions.  The parameter
     * iterate will be modified if the method is successful.
     *
     * @param function_value Value of the function at the initial point
     * @param iterate The initial point to begin the line search from
     * @param gradient The gradient at the initial point
     * @param search_direction A vector specifying the search direction
     * @param step_size Variable the calculated step size will be stored in
     *
     * @return false if no step size is suitable, true otherwise.
     */
    bool LineSearch_(double& function_value,
                     arma::mat& iterate,
                     arma::mat& gradient,
                     const arma::mat& search_direction,
                     double& step_size);

    /***
     * Find the L-BFGS search direction.
     *
     * @param gradient The gradient at the current point
     * @param iteration_num The iteration number
     * @param scaling_factor Scaling factor to use (see ChooseScalingFactor_())
     * @param search_direction Vector to store search direction in
     */
    void SearchDirection_(const arma::mat& gradient,
                          int iteration_num,
                          double scaling_factor,
                          arma::mat& search_direction);

    /***
     * Update the vectors y_bfgs_ and s_bfgs_, which store the differences
     * between the iterate and old iterate and the differences between the
     * gradient and the old gradient, respectively.
     *
     * @param iteration_num Iteration number
     * @param iterate Current point
     * @param old_iterate Point at last iteration
     * @param gradient Gradient at current point (iterate)
     * @param old_gradient Gradient at last iteration point (old_iterate)
     */
    void UpdateBasisSet_(int iteration_num,
                         const arma::mat& iterate,
                         const arma::mat& old_iterate,
                         const arma::mat& gradient,
                         const arma::mat& old_gradient);

  public:
    /***
     * Initialize the L-BFGS object.  Copy the function we will be optimizing
     * and set the size of the memory for the algorithm.
     *
     * @param function_in Instance of function to be optimized
     * @param num_basis Number of memory points to be stored
     */
    L_BFGS(FunctionType& function_in, int num_basis);

    /***
     * Return the point where the lowest function value has been found.
     *
     * @return arma::vec representing the point and a double with the function
     *     value at that point.
     */
    const std::pair<arma::mat, double>& min_point_iterate() const;

    /***
     * Set the maximum number of iterations for the line search.
     *
     * @param max_num_line_searches_in New maximum number of iterations
     */
    void set_max_num_line_searches(int max_num_line_searches_in);

    /***
     * Use L-BFGS to optimize the given function, starting at the given iterate
     * point and performing no more than the specified number of maximum
     * iterations.  The given starting point will be modified to store the
     * finishing point of the algorithm.
     *
     * @param num_iterations Maximum number of iterations to perform
     * @param iterate Starting point (will be modified)
     */
    bool Optimize(int num_iterations, arma::mat& iterate);
};

}; // namespace optimization
}; // namespace mlpack

#include "lbfgs_impl.h"

#endif
