/**
 * @file aug_lagrangian.hpp
 * @author Ryan Curtin
 *
 * Definition of AugLagrangian class, which implements the Augmented Lagrangian
 * optimization method (also called the 'method of multipliers'.  This class
 * uses the L-BFGS optimizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP
#define MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "aug_lagrangian_function.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AugLagrangian class implements the Augmented Lagrangian method of
 * optimization.  In this scheme, a penalty term is added to the Lagrangian.
 * This method is also called the "method of multipliers".
 *
 * The template class LagrangianFunctionType, used by the Optimize() method,
 * must implement the following five methods:
 *
 * - double Evaluate(const arma::mat& coordinates);
 * - void Gradient(const arma::mat& coordinates, arma::mat& gradient);
 * - size_t NumConstraints();
 * - double EvaluateConstraint(size_t index, const arma::mat& coordinates);
 * - double GradientConstraint(size_t index, const arma::mat& coordinates,
 *        arma::mat& gradient);
 *
 * The number of constraints must be greater than or equal to 0, and
 * EvaluateConstraint() should evaluate the constraint at the given index for
 * the given coordinates.  Evaluate() should provide the objective function
 * value for the given coordinates.
 */
class AugLagrangian
{
 public:
  /**
   * Initialize the Augmented Lagrangian with the default L-BFGS optimizer.  We
   * limit the number of L-BFGS iterations to 1000, rather than the unlimited
   * default L-BFGS.
   */
  AugLagrangian();

  /**
   * Initialize the Augmented Lagrangian with a custom L-BFGS optimizer.
   *
   * @param lbfgs The custom L-BFGS optimizer to be used.  This should have
   *    already been initialized with the given AugLagrangianFunction.
   */
  AugLagrangian(L_BFGS& lbfgs);

  /**
   * Optimize the function.  The value '1' is used for the initial value of each
   * Lagrange multiplier.  To set the Lagrange multipliers yourself, use the
   * other overload of Optimize().
   *
   * @tparam LagrangianFunctionType Function which can be optimized by this
   *     class.
   * @param function The function to optimize.
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param maxIterations Maximum number of iterations of the Augmented
   *     Lagrangian algorithm.  0 indicates no maximum.
   */
  template<typename LagrangianFunctionType>
  bool Optimize(LagrangianFunctionType& function,
                arma::mat& coordinates,
                const size_t maxIterations = 1000);

  /**
   * Optimize the function, giving initial estimates for the Lagrange
   * multipliers.  The vector of Lagrange multipliers will be modified to
   * contain the Lagrange multipliers of the final solution (if one is found).
   *
   * @tparam LagrangianFunctionType Function which can be optimized by this
   *      class.
   * @param function The function to optimize.
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param initLambda Vector of initial Lagrange multipliers.  Should have
   *     length equal to the number of constraints.
   * @param initSigma Initial penalty parameter.
   * @param maxIterations Maximum number of iterations of the Augmented
   *     Lagrangian algorithm.  0 indicates no maximum.
   */
  template<typename LagrangianFunctionType>
  bool Optimize(LagrangianFunctionType& function,
                arma::mat& coordinates,
                const arma::vec& initLambda,
                const double initSigma,
                const size_t maxIterations = 1000);

  //! Get the L-BFGS object used for the actual optimization.
  const L_BFGS& LBFGS() const { return lbfgs; }
  //! Modify the L-BFGS object used for the actual optimization.
  L_BFGS& LBFGS() { return lbfgs; }

  //! Get the Lagrange multipliers.
  const arma::vec& Lambda() const { return lambda; }
  //! Modify the Lagrange multipliers (i.e. set them before optimization).
  arma::vec& Lambda() { return lambda; }

  //! Get the penalty parameter.
  double Sigma() const { return sigma; }
  //! Modify the penalty parameter.
  double& Sigma() { return sigma; }

 private:
  //! If the user did not pass an L_BFGS object, we'll use our own internal one.
  L_BFGS lbfgsInternal;

  //! The L-BFGS optimizer that we will use.
  L_BFGS& lbfgs;

  //! Lagrange multipliers.
  arma::vec lambda;
  //! Penalty parameter.
  double sigma;

  /**
   * Internal optimization function: given an initialized AugLagrangianFunction,
   * perform the optimization itself.
   */
  template<typename LagrangianFunctionType>
  bool Optimize(AugLagrangianFunction<LagrangianFunctionType>& augfunc,
                arma::mat& coordinates,
                const size_t maxIterations);
};

} // namespace optimization
} // namespace mlpack

#include "aug_lagrangian_impl.hpp"

#endif // MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

