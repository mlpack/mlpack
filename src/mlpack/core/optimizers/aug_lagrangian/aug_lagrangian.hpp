/**
 * @file aug_lagrangian.hpp
 * @author Ryan Curtin
 *
 * Definition of AugLagrangian class, which implements the Augmented Lagrangian
 * optimization method (also called the 'method of multipliers'.  This class
 * uses the L-BFGS optimizer.
 */

#ifndef __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP
#define __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP

#include <mlpack/core.hpp>

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
  /**
   * Construct the Augmented Lagrangian optimizer with an instance of the given
   * function.
   *
   * @param function Function to be optimizer.
   * @param numBasis Number of points of memory for L-BFGS.
   */
  AugLagrangian(LagrangianFunction& function, size_t numBasis = 5);

  /**
   * Optimize the function.
   *
   * @param coordinates Output matrix to store the optimized coordinates in.
   * @param maxIterations Maximum number of iterations of the Augmented
   *     Lagrangian algorithm.  0 indicates no maximum.
   * @param sigma Initial penalty parameter.
   */
  bool Optimize(arma::mat& coordinates,
                const size_t maxIterations = 1000,
                double sigma = 0.5);

  //! Get the LagrangianFunction.
  const LagrangianFunction& Function() const { return function; }
  //! Modify the LagrangianFunction.
  LagrangianFunction& Function() { return function; }

  //! Get the number of memory points used by L-BFGS.
  size_t NumBasis() const { return numBasis; }
  //! Modify the number of memory points used by L-BFGS.
  size_t& NumBasis() { return numBasis; }

 private:
  //! Function to be optimized.
  LagrangianFunction& function;
  //! Number of memory points for L-BFGS.
  size_t numBasis;

  /**
   * This is a utility class, which we will pass to L-BFGS during the
   * optimization.  We use a utility class so that we do not have to expose
   * Evaluate() and Gradient() to the AugLagrangian public interface; instead,
   * with a private class, these methods are correctly protected (since they
   * should not be being used anywhere else).
   */
  class AugLagrangianFunction
  {
   public:
    AugLagrangianFunction(LagrangianFunction& functionIn,
                          arma::vec& lambdaIn,
                          double sigma);

    double Evaluate(const arma::mat& coordinates);
    void Gradient(const arma::mat& coordinates, arma::mat& gradient);

    const arma::mat& GetInitialPoint() const;

    //! Get the Lagrange multipliers.
    const arma::vec& Lambda() const { return lambda; }
    //! Modify the Lagrange multipliers.
    arma::vec& Lambda() { return lambda; }

    //! Get sigma.
    double Sigma() const { return sigma; }
    //! Modify sigma.
    double& Sigma() { return sigma; }

    //! Get the Lagrangian function.
    const LagrangianFunction& Function() const { return function; }
    //! Modify the Lagrangian function.
    LagrangianFunction& Function() { return function; }

   private:
    arma::vec lambda;
    double sigma;

    LagrangianFunction& function;
  };
};

}; // namespace optimization
}; // namespace mlpack

#include "aug_lagrangian_impl.hpp"

#endif // __MLPACK_CORE_OPTIMIZERS_AUG_LAGRANGIAN_AUG_LAGRANGIAN_HPP
