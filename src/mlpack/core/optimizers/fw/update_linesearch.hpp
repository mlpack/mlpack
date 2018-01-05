/**
 * @file update_linesearch.hpp
 * @author Chenzhe Diao
 *
 * Minimize convex function with line search, using secant method.
 * In FrankWolfe algorithm, used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_LINESEARCH_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_LINESEARCH_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/line_search/line_search.hpp>

namespace mlpack {
namespace optimization {

/**
 * Use line search in the update step for FrankWolfe algorithm. That is, take
 * \f$ \gamma = arg\min_{\gamma\in [0, 1]} f((1-\gamma)x + \gamma s) \f$.
 * The update rule would be:
 * \f[
 * x_{k+1} = (1-\gamma) x_k + \gamma s
 * \f]
 *
 */
class UpdateLineSearch
{
 public:
  /**
   * Construct the line search update rule.
   *
   * @param maxIter Max number of iterations in line search.
   * @param tolerance Tolerance for termination of line search.
   */
  UpdateLineSearch(const size_t maxIterations = 100000,
                   const double tolerance = 1e-5) :
      tolerance(tolerance), maxIterations(maxIterations)
  {/* Do nothing */}


  /**
   * Update rule for FrankWolfe, optimize with line search using secant method.
   *
   * FunctionType template parameters are required.
   * This class must implement the following functions:
   *
   * FunctionType:
   *
   *   double Evaluate(const arma::mat& coordinates);
   *   Evaluation of the function at specific coordinates.
   *
   *   void Gradient(const arma::mat& coordinates,
   *                 arma::mat& gradient);
   *   Solve the gradient of the function at specific coordinate,
   *   returned in gradient.
   *
   * @param function function to be optimized,
   * @param oldCoords previous solution coordinates, one end of line search.
   * @param s current linear_constr_solution result, the other end point of
   *        line search.
   * @param newCoords output new solution coords.
   * @param numIter current iteration number, not used here.
   */
  template<typename FunctionType>
  void Update(FunctionType& function,
              const arma::mat& oldCoords,
              const arma::mat& s,
              arma::mat& newCoords,
              const size_t /* numIter */)

  {
    LineSearch solver(maxIterations, tolerance);

    newCoords = s;
    solver.Optimize(function, oldCoords, newCoords);
  }

  //! Get the tolerance for termination.
  double Tolerance() const {return tolerance;}
  //! Modify the tolerance for termination.
  double& Tolerance() {return tolerance;}

  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIterations; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIterations; }

 private:
  //! Tolerance for convergence.
  double tolerance;

  //! Max number of iterations.
  size_t maxIterations;
};  // class UpdateLineSearch

} // namespace optimization
} // namespace mlpack

#endif
