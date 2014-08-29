/**
 * @file aug_lagrangian.hpp
 * @author Ryan Curtin
 *
 * Definition of AugLagrangian class, which implements the Augmented Lagrangian
 * optimization method (also called the 'method of multipliers'.  This class
 * uses the L-BFGS optimizer.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP
#define __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/lbfgs/lbfgs.hpp>

#include "aug_lagrangian_function.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AugLagrangian class implements the Augmented Lagrangian method of
 * optimization.  In this scheme, a penalty term is added to the Lagrangian.
 * This method is also called the "method of multipliers".
 *
 * The template class LagrangianFunction must implement the following five
 * methods:
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
 *
 * @tparam LagrangianFunction Function which can be optimized by this class.
 */
template<typename LagrangianFunction>
class AugLagrangian
{
 public:
  //! Shorthand for the type of the L-BFGS optimizer we'll be using.
  typedef L_BFGS<AugLagrangianFunction<LagrangianFunction> >
      L_BFGSType;

  /**
   * Initialize the Augmented Lagrangian with the default L-BFGS optimizer.  We
   * limit the number of L-BFGS iterations to 1000, rather than the unlimited
   * default L-BFGS.
   *
   * @param function The function to be optimized.
   */
  AugLagrangian(LagrangianFunction& function);

  /**
   * Initialize the Augmented Lagrangian with a custom L-BFGS optimizer.
   *
   * @param function The function to be optimized.  This must be a pre-created
   *    utility AugLagrangianFunction.
   * @param lbfgs The custom L-BFGS optimizer to be used.  This should have
   *    already been initialized with the given AugLagrangianFunction.
   */
  AugLagrangian(AugLagrangianFunction<LagrangianFunction>& augfunc,
                L_BFGSType& lbfgs);

  /**
   * Optimize the function.  The value '1' is used for the initial value of each
   * Lagrange multiplier.  To set the Lagrange multipliers yourself, use the
   * other overload of Optimize().
   *
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param maxIterations Maximum number of iterations of the Augmented
   *     Lagrangian algorithm.  0 indicates no maximum.
   * @param sigma Initial penalty parameter.
   */
  bool Optimize(arma::mat& coordinates,
                const size_t maxIterations = 1000);

  /**
   * Optimize the function, giving initial estimates for the Lagrange
   * multipliers.  The vector of Lagrange multipliers will be modified to
   * contain the Lagrange multipliers of the final solution (if one is found).
   *
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param initLambda Vector of initial Lagrange multipliers.  Should have
   *     length equal to the number of constraints.
   * @param initSigma Initial penalty parameter.
   * @param maxIterations Maximum number of iterations of the Augmented
   *     Lagrangian algorithm.  0 indicates no maximum.
   */
  bool Optimize(arma::mat& coordinates,
                const arma::vec& initLambda,
                const double initSigma,
                const size_t maxIterations = 1000);

  //! Get the LagrangianFunction.
  const LagrangianFunction& Function() const { return function; }
  //! Modify the LagrangianFunction.
  LagrangianFunction& Function() { return function; }

  //! Get the L-BFGS object used for the actual optimization.
  const L_BFGSType& LBFGS() const { return lbfgs; }
  //! Modify the L-BFGS object used for the actual optimization.
  L_BFGSType& LBFGS() { return lbfgs; }

  //! Get the Lagrange multipliers.
  const arma::vec& Lambda() const { return augfunc.Lambda(); }
  //! Modify the Lagrange multipliers (i.e. set them before optimization).
  arma::vec& Lambda() { return augfunc.Lambda(); }

  //! Get the penalty parameter.
  double Sigma() const { return augfunc.Sigma(); }
  //! Modify the penalty parameter.
  double& Sigma() { return augfunc.Sigma(); }

  // convert the obkect into a string
  std::string ToString() const;

 private:
  //! Function to be optimized.
  LagrangianFunction& function;

  //! Internally used AugLagrangianFunction which holds the function we are
  //! optimizing.  This isn't publically accessible, but we provide ways to get
  //! to the Lagrange multipliers and the penalty parameter sigma.
  AugLagrangianFunction<LagrangianFunction> augfunc;

  //! If the user did not pass an L_BFGS object, we'll use our own internal one.
  L_BFGSType lbfgsInternal;

  //! The L-BFGS optimizer that we will use.
  L_BFGSType& lbfgs;
};

}; // namespace optimization
}; // namespace mlpack

#include "aug_lagrangian_impl.hpp"

#endif // __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

