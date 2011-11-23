/**
 * @file lbfgs_impl.hpp
 * @author Dongryeol Lee (dongryel@cc.gatech.edu)
 * @author Ryan Curtin
 *
 * The implementation of the L_BFGS optimizer.
 */
#ifndef __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP
#define __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP

namespace mlpack {
namespace optimization {

/**
 * Evaluate the function at the given iterate point and store the result if
 * it is a new minimum.
 *
 * @return The value of the function
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::Evaluate_(const arma::mat& iterate)
{
  // Evaluate the function and keep track of the minimum function
  // value encountered during the optimization.
  double function_value = function_.Evaluate(iterate);

  if (function_value < min_point_iterate_.second)
  {
    min_point_iterate_.first = iterate;
    min_point_iterate_.second = function_value;
  }

  return function_value;
}

/**
 * Calculate the scaling factor gamma which is used to scale the Hessian
 * approximation matrix.  See method M3 in Section 4 of Liu and Nocedal (1989).
 *
 * @return The calculated scaling factor
 */
template<typename FunctionType>
double L_BFGS<FunctionType>::ChooseScalingFactor_(int iteration_num,
                                                  const arma::mat& gradient)
{
  double scaling_factor = 1.0;
  if (iteration_num > 0)
  {
    int previous_pos = (iteration_num - 1) % num_basis_;
    // Get s and y matrices once instead of multiple times.
    arma::mat& s_col = s_lbfgs_.slice(previous_pos);
    arma::mat& y_col = y_lbfgs_.slice(previous_pos);
    scaling_factor = dot(s_col, y_col) / dot(y_col, y_col);
  }
  else
  {
    scaling_factor = 1.0 / sqrt(dot(gradient, gradient));
  }

  return scaling_factor;
}

/**
 * Check to make sure that the norm of the gradient is not smaller than 1e-10.
 * Currently that value is not configurable.
 *
 * @return (norm < lbfgs/min_gradient_norm)
 */
template<typename FunctionType>
bool L_BFGS<FunctionType>::GradientNormTooSmall_(const arma::mat& gradient)
{
  double norm = arma::norm(gradient, 2);

  return (norm < CLI::GetParam<double>("lbfgs/min_gradient_norm"));
}

/**
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
                                       arma::mat& iterate,
                                       arma::mat& gradient,
                                       const arma::mat& search_direction,
                                       double& step_size)
{
  // The initial linear term approximation in the direction of the
  // search direction.
  double initial_search_direction_dot_gradient =
      arma::dot(gradient, search_direction);

  // If it is not a descent direction, just report failure.
  if (initial_search_direction_dot_gradient > 0.0)
    return false;

  // Save the initial function value.
  double initial_function_value = function_value;

  // Unit linear approximation to the decrease in function value.
  double linear_approx_function_value_decrease =
      CLI::GetParam<double>("lbfgs/armijo_constant") *
      initial_search_direction_dot_gradient;

  // The number of iteration in the search.
  int num_iterations = 0;

  // Armijo step size scaling factor for increase and decrease.
  const double inc = 2.1;
  const double dec = 0.5;
  double width = 0;

  while(true)
  {
    // Perform a step and evaluate the gradient and the function values at that
    // point.
    new_iterate_tmp_ = iterate;
    new_iterate_tmp_ += step_size * search_direction;
    function_value = Evaluate_(new_iterate_tmp_);
    function_.Gradient(new_iterate_tmp_, gradient);
    num_iterations++;

    if (function_value > initial_function_value + step_size *
        linear_approx_function_value_decrease)
    {
      width = dec;
    }
    else
    {
      // Check Wolfe's condition.
      double search_direction_dot_gradient =
          arma::dot(gradient, search_direction);
      double wolfe = CLI::GetParam<double>("lbfgs/wolfe");

      if(search_direction_dot_gradient < wolfe *
          initial_search_direction_dot_gradient)
      {
        width = inc;
      }
      else
      {
        if (search_direction_dot_gradient > -wolfe *
            initial_search_direction_dot_gradient)
        {
          width = dec;
        }
        else
        {
          break;
        }
      }
    }

    // Terminate when the step size gets too small or too big or it
    // exceeds the max number of iterations.
    if ((step_size < CLI::GetParam<double>("lbfgs/min_step")) ||
        (step_size > CLI::GetParam<double>("lbfgs/max_step")) ||
        (num_iterations >= CLI::GetParam<int>("lbfgs/max_line_search_trials")))
    {
      return false;
    }

    // Scale the step size.
    step_size *= width;
  }

  // Move to the new iterate.
  iterate = new_iterate_tmp_;
  return true;
}

/**
 * Find the L_BFGS search direction.
 *
 * @param gradient The gradient at the current point
 * @param iteration_num The iteration number
 * @param scaling_factor Scaling factor to use (see ChooseScalingFactor_())
 * @param search_direction Vector to store search direction in
 */
template<typename FunctionType>
void L_BFGS<FunctionType>::SearchDirection_(const arma::mat& gradient,
                                            int iteration_num,
                                            double scaling_factor,
                                            arma::mat& search_direction)
{
  arma::mat q = gradient;

  // See "A Recursive Formula to Compute H * g" in "Updating quasi-Newton
  // matrices with limited storage" (Nocedal, 1980).

  // Temporary variables.
  arma::vec rho(num_basis_);
  arma::vec alpha(num_basis_);

  int limit = std::max(iteration_num - num_basis_, 0);
  for (int i = iteration_num - 1; i >= limit; i--)
  {
    int translated_position = i % num_basis_;
    rho[iteration_num - i - 1] = 1.0 / arma::dot(
        y_lbfgs_.slice(translated_position),
        s_lbfgs_.slice(translated_position));
    alpha[iteration_num - i - 1] = rho[iteration_num - i - 1] *
        arma::dot(s_lbfgs_.slice(translated_position), q);
    q -= alpha[iteration_num - i - 1] * y_lbfgs_.slice(translated_position);
  }

  search_direction = scaling_factor * q;

  for (int i = limit; i <= iteration_num - 1; i++)
  {
    int translated_position = i % num_basis_;
    double beta = rho[iteration_num - i - 1] *
        arma::dot(y_lbfgs_.slice(translated_position), search_direction);
    search_direction += (alpha[iteration_num - i - 1] - beta) *
        s_lbfgs_.slice(translated_position);
  }

  // Negate the search direction so that it is a descent direction.
  search_direction *= -1;
}

/**
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
                                           const arma::mat& iterate,
                                           const arma::mat& old_iterate,
                                           const arma::mat& gradient,
                                           const arma::mat& old_gradient)
{
  // Overwrite a certain position instead of pushing everything in the vector
  // back one position
  int overwrite_pos = iteration_num % num_basis_;
  s_lbfgs_.slice(overwrite_pos) = iterate - old_iterate;
  y_lbfgs_.slice(overwrite_pos) = gradient - old_gradient;
}

/***
 * Initialize the L_BFGS object.  Copy the function we will be optimizing and
 * set the size of the memory for the algorithm.
 *
 * @param function_in Instance of function to be optimized
 * @param num_basis Number of memory points to be stored
 */
template<typename FunctionType>
L_BFGS<FunctionType>::L_BFGS(FunctionType& function_in, int num_basis) :
  function_(function_in)
{
  // Get the dimensions of the coordinates of the function; GetInitialPoint()
  // might return an arma::vec, but that's okay because then n_cols will simply
  // be 1.
  int rows = function_.GetInitialPoint().n_rows;
  int cols = function_.GetInitialPoint().n_cols;

  new_iterate_tmp_.set_size(rows, cols);
  s_lbfgs_.set_size(rows, cols, num_basis);
  y_lbfgs_.set_size(rows, cols, num_basis);
  num_basis_ = num_basis;

  // Allocate the pair holding the min iterate information.
  min_point_iterate_.first.zeros(rows, cols);
  min_point_iterate_.second = std::numeric_limits<double>::max();
}

/**
 * Return the point where the lowest function value has been found.
 *
 * @return arma::vec representing the point and a double with the function
 *     value at that point.
 */
template<typename FunctionType>
const std::pair<arma::mat, double>&
L_BFGS<FunctionType>::min_point_iterate() const
{
  return min_point_iterate_;
}

/**
 * Use L_BFGS to optimize the given function, starting at the given iterate
 * point and performing no more than the specified number of maximum iterations.
 * The given starting point will be modified to store the finishing point of the
 * algorithm.
 *
 * @param num_iterations Maximum number of iterations to perform
 * @param iterate Starting point (will be modified)
 */
template<typename FunctionType>
bool L_BFGS<FunctionType>::Optimize(int num_iterations, arma::mat& iterate)
{
  // The old iterate to be saved.
  arma::mat old_iterate;
  old_iterate.zeros(iterate.n_rows, iterate.n_cols);

  // Whether to optimize until convergence.
  bool optimize_until_convergence = (num_iterations <= 0);

  // The initial function value.
  double function_value = Evaluate_(iterate);

  // The gradient: the current and the old.
  arma::mat gradient;
  arma::mat old_gradient;
  gradient.zeros(iterate.n_rows, iterate.n_cols);
  old_gradient.zeros(iterate.n_rows, iterate.n_cols);

  // The search direction.
  arma::mat search_direction;
  search_direction.zeros(iterate.n_rows, iterate.n_cols);

  // The initial gradient value.
  function_.Gradient(iterate, gradient);

  // The flag denoting whether or not the optimization has been successful.
  bool success = false;

  // The main optimization loop.
  for (int it_num = 0; optimize_until_convergence || it_num < num_iterations;
       it_num++)
  {
    Log::Debug << "L-BFGS iteration " << it_num << "; objective " <<
        function_.Evaluate(iterate) << "." << std::endl;

    // Break when the norm of the gradient becomes too small.
    if(GradientNormTooSmall_(gradient))
    {
      success = true; // We have found the minimum.
      Log::Debug << "L-BFGS gradient norm too small (terminating)."
          << std::endl;
      break;
    }

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
    success = LineSearch_(function_value, iterate, gradient, search_direction,
        step_size);

    if (!success)
      break; // The line search failed; nothing else to try.

    // Overwrite an old basis set.
    UpdateBasisSet_(it_num, iterate, old_iterate, gradient, old_gradient);

  } // end of the optimization loop.

  return success;
}

}; // namespace optimization
}; // namespace mlpack

#endif // __MLPACK_CORE_OPTIMIZERS_LBFGS_LBFGS_IMPL_HPP
