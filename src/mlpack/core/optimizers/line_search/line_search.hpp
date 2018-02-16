/**
 * @file line_search.hpp
 * @author Chenzhe Diao
 *
 * Minimize a function using line search with secant method.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_LINE_SEARCH_LINE_SEARCH_HPP
#define MLPACK_CORE_OPTIMIZERS_LINE_SEARCH_LINE_SEARCH_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

/**
 * Find the minimum of a function along the line between two points.
 * The solver uses the secant method to find the zero of the derivative of the
 * function along the search line.
 *
 * If the function is convex, the derivative of the function along the search
 * line will be nondecreasing, so the minimum always exists.
 * If the function is strongly convex, the derivative of the function along the
 * search line will be strictly increasing, so the minimum is unique.
 */
class LineSearch
{
 public:
  LineSearch(const size_t maxIterations = 100000,
             const double tolerance = 1e-5) :
      maxIterations(maxIterations), tolerance(tolerance)
  {/* Do nothing */ }

  /**
   * Line search to minimize function between two points with Secant method,
   * that is, to find the zero of Derivative(gamma), where gamma is in [0,1].
   *
   * The function is assumed to be convex here, otherwise might not converge.
   *
   * @param function function to be minimized.
   * @param x1 Input one end point.
   * @param x2 Input the other end point, also used as output, to store the
   *           coordinate of the optimal solution.
   * @return Minimum solution function value.
   */
  template<typename FunctionType>
  double Optimize(FunctionType& function, const arma::mat& x1, arma::mat& x2);

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

  //! Get the tolerance for termination.
  double Tolerance() const { return tolerance; }
  //! Modify the tolerance for termination.
  double& Tolerance() { return tolerance; }

 private:
  //! Max number of iterations.
  size_t maxIterations;

  //! Tolerance for convergence.
  double tolerance;

  /**
   * Derivative of the function along the search line.
   *
   * @param function original function.
   * @param x0 starting point.
   * @param deltaX distance between two end points.
   * @param gamma position of the point in the search line, take in [0, 1].
   *
   * @return Derivative of function(x0 + gamma * deltaX) with respect to gamma.
   */
  template<typename FunctionType>
  double Derivative(FunctionType& function,
                    const arma::mat& x0,
                    const arma::mat& deltaX,
                    const double gamma);
};  // class LineSearch
} // namespace optimization
} // namespace mlpack

#include "line_search_impl.hpp"

#endif
