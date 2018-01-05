/**
 * @file update_full_correction.hpp
 * @author Chenzhe Diao
 *
 * Update method for FrankWolfe algorithm, recalculate the coefficents of
 * of current atoms, while satisfying the norm constraint.
 * Used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_FULL_CORRECTION_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_FULL_CORRECTION_HPP

#include <mlpack/prereqs.hpp>
#include "atoms.hpp"

namespace mlpack {
namespace optimization {

/**
 * Full correction approach to update the solution.
 *
 * UpdateSpan class reoptimize the solution in the span of all current atoms,
 * which is used in OMP, which only focus on sparsity.
 *
 * UpdateFullCorrection class reoptimize the solution in a similar way, however,
 * the solutions need to satisfy the constraint that the atom norm has to be
 * smaller than or equal to tau. This constraint optimization problem is solved
 * by projected gradient method. See Atoms.ProjectedEnhancement().
 *
 * Currently only works for function in FuncSq class.
 *
 */
class UpdateFullCorrection
{
 public:
  /**
   * Construct UpdateFullCorrection class.
   *
   * @param tau atom norm constraint.
   * @param stepSize step size used in projected gradient method.
   */
  UpdateFullCorrection(const double tau, const double stepSize) :
      tau(tau), stepSize(stepSize)
  { /* Do nothing. */ }

  /**
   * Update rule for FrankWolfe, recalculate the coefficents of of current
   * atoms, while satisfying the norm constraint.
   *
   * @param function function to be optimized.
   * @param oldCoords previous solution coords.
   * @param s current linear_constr_solution result.
   * @param newCoords new output solution coords.
   * @param numIter current iteration number.
   */
  void Update(FuncSq& function,
              const arma::mat& oldCoords,
              const arma::mat& s,
              arma::mat& newCoords,
              const size_t /* numIter */)
  {
    // Line search, with explicit solution here.
    arma::mat v = tau * s - oldCoords;
    arma::mat b = function.Vectorb();
    arma::mat A = function.MatrixA();
    double gamma = arma::dot(b - A * oldCoords, A * v);
    gamma = gamma / std::pow(arma::norm(A * v, "fro"), 2);
    gamma = std::min(gamma, 1.0);
    atoms.CurrentCoeffs() = (1.0 - gamma) * atoms.CurrentCoeffs();
    atoms.AddAtom(s, function, gamma * tau);

    // Projected gradient method for enhancement.
    atoms.ProjectedGradientEnhancement(function, tau, stepSize);
    atoms.RecoverVector(newCoords);
  }

 private:
  //! Atom norm constraint.
  double tau;

  //! Step size in projected gradient method.
  double stepSize;

  //! Atoms information.
  Atoms atoms;
};

} // namespace optimization
} // namespace mlpack

#endif
