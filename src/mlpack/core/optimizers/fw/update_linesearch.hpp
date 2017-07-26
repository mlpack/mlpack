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
  UpdateLineSearch(const size_t maxIter = 100000,
                   const double tolerance = 1e-5) :
      maxIter(maxIter), tolerance(tolerance)
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
              const size_t numIter)

  {
    double gamma = LineSearchSecant(function, oldCoords, s);
    newCoords = (1 - gamma) * oldCoords + gamma * s;
  }

  //! Get the tolerance for termination.
  double Tolerance() const {return tolerance;}
  //! Modify the tolerance for termination.
  double& Tolerance() {return tolerance;}
  
  //! Get the maximum number of iterations (0 indicates no limit).
  size_t MaxIterations() const { return maxIter; }
  //! Modify the maximum number of iterations (0 indicates no limit).
  size_t& MaxIterations() { return maxIter; }

 private:
  //! Tolerance for convergence.
  double tolerance;
  
  //! Max number of iterations.
  size_t maxIter;

  /**
   * Line search to minimize function between two points with Secant method,
   * that is, to find the zero of Derivative(gamma), where gamma is in [0,1].
   *
   * The function is assumed to be convex here, otherwise might not converge.
   *
   * If the function is convex, Derivative(gamma) would be a nondecreasing
   * function of gamma, then the soluton exists.
   * If the function is strongly convex, Derivative(gamma) is strictly
   * increasing, then the solution exists and is unique.
   *
   * @param function function to be minimized.
   * @param x1 One end point.
   * @param x2 The other end point.
   * @return Optimal solution position ratio betwen two points,
   *         0 means x1, 1 means x2.
   */
  template<typename FunctionType>
  double LineSearchSecant(const FunctionType& function,
                          const arma::mat& x1,
                          const arma::mat& x2)
  {
    // Set up the search line, that is,
    // find the zero of der(gamma) = Derivative(gamma).
    arma::mat deltaX = x2 - x1;
    double gamma = 0;
    double derivative = Derivative(function, x1, deltaX, 0);
    double derivativeNew = Derivative(function, x1, deltaX, 1);
    double secant = derivativeNew - derivative;
    
    if (derivative >= 0.0) // Optimal solution at left endpoint.
      return 0.0;
    else if (derivativeNew <= 0.0) // Optimal solution at righ endpoint.
      return 1.0;
    else if (secant < tolerance) // function too flat, just take left endpoint.
      return 0.0;
    
    // Line search by Secant Method.
    for (size_t k = 0; k < maxIter; ++k)
    {
      // secant should always >=0 for convex function.
      if (secant < 0.0)
      {
        Log::Fatal << "LineSearchSecant: Function is not convex!" << std::endl;
        return 0.0;
      }
      
      // Solve new gamma.
      double gammaNew = gamma - derivative / secant;
      gammaNew = std::max(gammaNew, 0.0);
      gammaNew = std::min(gammaNew, 1.0);
      
      // Update secant, gamma and derivative
      derivativeNew = Derivative(function, x1, deltaX, gammaNew);
      secant = (derivativeNew - derivative) / (gammaNew - gamma);
      gamma = gammaNew;
      derivative = derivativeNew;
      
      if(std::fabs(derivative) < tolerance)
      {
        Log::Info << "LineSearchSecant: minimized within tolerance "
            << tolerance << "; " << "terminating optimization." << std::endl;
        return gamma;
      }
    }

    Log::Info << "LineSearchSecant: maximum iterations (" << maxIter
        << ") reached; " << "terminating optimization." << std::endl;

    return gamma;
  }  // LineSearchSecant()
  
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
  double Derivative(const FunctionType& function,
                    const arma::mat& x0,
                    const arma::mat& deltaX,
                    const double gamma)
  {
    arma::mat gradient;
    function.Gradient(x0 + gamma * deltaX, gradient);
    return arma::dot(gradient, deltaX);
  }
};  // class UpdateLineSearch

} // namespace optimization
} // namespace mlpack

#endif
