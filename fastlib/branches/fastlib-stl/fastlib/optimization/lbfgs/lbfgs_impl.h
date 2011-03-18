/** @author Dongryeol Lee (dongryel@cc.gatech.edu)
 *
 *  @brief The implementation of the L_BFGS optimizer.
 *
 *  @file lbfgs_impl.h
 */

#ifndef OPTIMIZATION_LBFGS_IMPL_H
#define OPTIMIZATION_LBFGS_IMPL_H

namespace mlpack {
namespace optimization {

template<typename FunctionType>
void L_BFGS<FunctionType>::LbfgsParam::set_max_num_line_searches(
    int max_num_line_searches_in) {
  max_line_search_ = max_num_line_searches_in;
}

template<typename FunctionType>
double L_BFGS<FunctionType>::LbfgsParam::armijo_constant() const {
  return armijo_constant_;
}

template<typename FunctionType>
double L_BFGS<FunctionType>::LbfgsParam::min_step() const {
  return min_step_;
}

template<typename FunctionType>
double L_BFGS<FunctionType>::LbfgsParam::max_step() const {
  return max_step_;
}

template<typename FunctionType>
int L_BFGS<FunctionType>::LbfgsParam::max_line_search() const {
  return max_line_search_;
}

template<typename FunctionType>
double L_BFGS<FunctionType>::LbfgsParam::wolfe() const {
  return wolfe_;
}

template<typename FunctionType>
L_BFGS<FunctionType>::LbfgsParam::LbfgsParam() {
  armijo_constant_ = 1e-4;
  min_step_ = 1e-20;
  max_step_ = 1e20;
  max_line_search_ = 20;
  wolfe_ = 0.9;
}

/***
 * Evaluate the function at the given iterate point and store the result if
 * it is a new minimum.
 *
 * @return The value of the function
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::Evaluate_(const arma::vec& iterate) {

  // Evaluate the function and keep track of the minimum function
  // value encountered during the optimization.
  double function_value = function_->Evaluate(iterate);

  if(function_value < min_point_iterate_.second) {
    min_point_iterate_.first = iterate;
    min_point_iterate_.second = function_value;
  }

  return function_value;
}

/***
 * Calculate the scaling factor gamma which is used to scale the Hessian
 * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal (1989).
 *
 * @return The calculated scaling factor
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::ChooseScalingFactor_(int iteration_num,
                                                  const arma::vec& gradient) {
  double scaling_factor = 1.0;
  if(iteration_num > 0) {
    int previous_pos = (iteration_num - 1) % num_basis_;
    // Get column subviews once instead of several times.
    arma::subview_col<double> s_col = s_lbfgs_.col(previous_pos);
    arma::subview_col<double> y_col = y_lbfgs_.col(previous_pos);
    scaling_factor = dot(s_col, y_col) / dot(y_col, y_col);
  } else {
    scaling_factor = 1.0 / sqrt(dot(gradient, gradient));
  }

  return scaling_factor;
}

/***
 * Check to make sure that the norm of the gradient is not smaller than 1e-5.
 * Currently that value is not configurable.
 *
 * @return (norm < 1e-5)
 */
template<typename FunctionType>
bool L_BFGS<FunctionType>::GradientNormTooSmall_(const arma::vec &gradient) {
  const double threshold = 1e-5; // TODO: this threshold should be configurable
  return arma::norm(gradient, 2) < threshold;
}

/***
 * Perform a back-tracking line search along the search direction to calculate a
 * step size satisfying the Wolfe conditions.
 *
 * @param function_value Value of the function at the initial point
 * @param iterate The initial point to begin the line search from
 * @param gradient The gradient at the initial point
 * @param search_direction A vector specifying the search direction
 * @param step_size Variable the calculated step size will be stored in
 *
 * @return false if no step size is suitable, true otherwise.
 */
template<typename FunctionType>
bool L_BFGS<FunctionType>::LineSearch_(double& function_value,
                                       arma::vec& iterate,
                                       arma::vec& gradient,
                                       const arma::vec& search_direction,
                                       double& step_size) {
  // Implements the line search with back-tracking.

  // The initial linear term approximation in the direction of the
  // search direction.
  double initial_search_direction_dot_gradient =
    arma::dot(gradient, search_direction);

  // If it is not a descent direction, just report failure.
  if(initial_search_direction_dot_gradient > 0.0)
    return false;

  // Save the initial function value.
  double initial_function_value = function_value;

  // Unit linear approximation to the decrease in function value.
  double linear_approx_function_value_decrease = param_.armijo_constant() *
      initial_search_direction_dot_gradient;

  // The number of iteration in the search.
  int num_iterations = 0;

  // Armijo step size scaling factor for increase and decrease.
  const double inc = 2.1;
  const double dec = 0.5;
  double width = 0;
  for(; ;) {
    // Perform a step and evaluate the gradient and the function
    // values at that point.
    new_iterate_tmp_ = iterate;
    new_iterate_tmp_ += step_size * search_direction;
    function_value = Evaluate_(new_iterate_tmp_);
    function_->Gradient(new_iterate_tmp_, gradient);
    num_iterations++;

    if(function_value > initial_function_value + step_size *
        linear_approx_function_value_decrease) {
      width = dec;
    } else {
      // Check Wolfe's condition.
      double search_direction_dot_gradient =
          arma::dot(gradient, search_direction);

      if(search_direction_dot_gradient < param_.wolfe() *
          initial_search_direction_dot_gradient) {
        width = inc;
      } else {
        if(search_direction_dot_gradient > -param_.wolfe() *
            initial_search_direction_dot_gradient) {
          width = dec;
        } else {
          break;
        }
      }
    }

    // Terminate when the step size gets too small or too big or it
    // exceeds the max number of iterations.
    if((step_size < param_.min_step()) || 
       (step_size > param_.max_step()) ||
       (num_iterations >= param_.max_line_search())) {
      return false;
    }

    // Scale the step size.
    step_size *= width;
  }

  // Move to the new iterate.
  iterate = new_iterate_tmp_;
  return true;
}

/***
 * Find the L_BFGS search direction.
 *
 * @param gradient The gradient at the current point
 * @param iteration_num The iteration number
 * @param scaling_factor Scaling factor to use (see ChooseScalingFactor_())
 * @param search_direction Vector to store search direction in
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::SearchDirection_(const arma::vec& gradient,
                                            int iteration_num,
                                            double scaling_factor,
                                            arma::vec& search_direction) {
  arma::vec q = gradient;

  // Temporary variables.
  arma::vec rho(num_basis_);
  arma::vec alpha(num_basis_);

  int limit = std::max(iteration_num - num_basis_, 0);
  for(int i = iteration_num - 1; i >= limit; i--) {
    int translated_position = i % num_basis_;
    rho[iteration_num - i - 1] = 1.0 / arma::dot(
        y_lbfgs_.col(translated_position),
        s_lbfgs_.col(translated_position));
    alpha[iteration_num - i - 1] = rho[iteration_num - i - 1] *
        arma::dot(s_lbfgs_.col(translated_position), q);
  }
  search_direction = scaling_factor * q;
  for(int i = limit; i <= iteration_num - 1; i++) {
    int translated_position = i % num_basis_;
    double beta = rho[iteration_num - i - 1] *
        arma::dot(y_lbfgs_.col(translated_position), search_direction);
    search_direction += (alpha[iteration_num - i - 1] - beta) *
        s_lbfgs_.col(translated_position);
  }

  // Negate the search direction so that it is a descent direction.
  for(unsigned int i = 0; i < search_direction.n_elem; i++)
    search_direction[i] = -search_direction[i];
}

/***
 * Update the vectors y_bfgs_ and s_bfgs_, which store the differences between
 * the iterate and old iterate and the differences between the gradient and the
 * old gradient, respectively.
 *
 * @param iteration_num Iteration number
 * @param iterate Current point
 * @param old_iterate Point at last iteration
 * @param gradient Gradient at current point (iterate)
 * @param old_gradient Gradient at last iteration point (old_iterate)
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::UpdateBasisSet_(int iteration_num,
                                           const arma::vec& iterate,
                                           const arma::vec& old_iterate,
                                           const arma::vec& gradient,
                                           const arma::vec& old_gradient) {
  // Overwrite a certain position instead of pushing everything in the vector
  // back one position
  int overwrite_pos = iteration_num % num_basis_;
  s_lbfgs_.col(overwrite_pos) = iterate - old_iterate;
  y_lbfgs_.col(overwrite_pos) = gradient - old_gradient;
}

/***
 * Initialize the L_BFGS object.  Copy the function we will be optimizing and
 * set the size of the memory for the algorithm.
 *
 * @param function_in Instance of function to be optimized
 * @param num_basis Number of memory points to be stored
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::Init(FunctionType& function_in, int num_basis) {
  function_ = &function_in;
  new_iterate_tmp_.set_size(function_->GetDimension());
  s_lbfgs_.set_size(function_->GetDimension(), num_basis);
  y_lbfgs_.set_size(function_->GetDimension(), num_basis);
  num_basis_ = num_basis;

  // Allocate the pair holding the min iterate information.
  min_point_iterate_.first.set_size(function_->GetDimension());
  min_point_iterate_.first.zeros();
  min_point_iterate_.second = std::numeric_limits<double>::max();
}

/***
 * Return the point where the lowest function value has been found.
 *
 * @return arma::vec representing the point and a double with the function
 *     value at that point.
 */
template<typename FunctionType>
const std::pair<arma::vec, double>&
L_BFGS<FunctionType>::min_point_iterate() const {
  return min_point_iterate_;
}

/***
 * Set the maximum number of iterations for the line search.
 *                                                                                                                                                                                              
 * @param max_num_line_searches_in New maximum number of iterations
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::set_max_num_line_searches(
    int max_num_line_searches_in) {
  param_.set_max_num_line_searches(max_num_line_searches_in);
}

/***
 * Use L_BFGS to optimize the given function, starting at the given iterate
 * point and performing no more than the specified number of maximum iterations.
 * The given starting point will be modified to store the finishing point of the
 * algorithm.
 *
 * @param num_iterations Maximum number of iterations to perform
 * @param iterate Starting point (will be modified)
 */
template<typename FunctionType>
bool L_BFGS<FunctionType>::Optimize(int num_iterations, arma::vec& iterate) {
  // The old iterate to be saved.
  arma::vec old_iterate;
  old_iterate.set_size(function_->GetDimension());
  old_iterate.zeros();

  // Whether to optimize until convergence.
  bool optimize_until_convergence = (num_iterations <= 0);

  // The initial function value.
  double function_value = Evaluate_(iterate);

  // The gradient: the current and the old.
  arma::vec gradient;
  arma::vec old_gradient;
  gradient.set_size(function_->GetDimension());
  gradient.zeros();
  old_gradient.set_size(function_->GetDimension());
  old_gradient.zeros();

  // The search direction.
  arma::vec search_direction;
  search_direction.set_size(function_->GetDimension());
  search_direction.zeros();

  // The initial gradient value.
  function_->Gradient(iterate, gradient);

  // The boolean flag telling whether the line search succeeded at
  // least once.
  bool line_search_successful_at_least_once = false;

  // The main optimization loop.
  for(int it_num = 0; optimize_until_convergence || it_num < num_iterations;
      it_num++) {

    // Break when the norm of the gradient becomes too small.
    if(GradientNormTooSmall_(gradient))
      break;

    // Choose the scaling factor.
    double scaling_factor = ChooseScalingFactor_(it_num, gradient);

    // Build an approximation to the Hessian and choose the search
    // direction for the current iteration.
    SearchDirection_(gradient, it_num, scaling_factor, search_direction);

    // Save the old iterate and the gradient before stepping.
    old_iterate = iterate;
    old_gradient = gradient;

    // Do a line search and take a step.
    double step_size = 1.0;
    bool search_is_success = LineSearch_(function_value, iterate, gradient,
        search_direction, step_size);

    if(!search_is_success)
      break;

    line_search_successful_at_least_once = search_is_success;

    // Overwrite an old basis set.
    UpdateBasisSet_(it_num, iterate, old_iterate, gradient, old_gradient);

  } // end of the optimization loop.

  return line_search_successful_at_least_once;
}

}; // namespace optimization
}; // namespace mlpack

#endif
