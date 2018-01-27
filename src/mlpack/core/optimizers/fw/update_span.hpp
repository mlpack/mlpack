/**
 * @file update_span.hpp
 * @author Chenzhe Diao
 *
 * Update method for FrankWolfe algorithm, recalculate the optimal in the span
 * of previous solution space. Used as UpdateRuleType.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_UPDATE_SPAN_HPP

#include <mlpack/prereqs.hpp>
#include "func_sq.hpp"
#include "atoms.hpp"

namespace mlpack {
namespace optimization {

/**
 * Recalculate the optimal solution in the span of all previous solution space,
 * used as update step for FrankWolfe algorithm.
 *
 * Currently only works for function in FuncSq class.
 */
class UpdateSpan
{
 public:
  /**
   * Construct the span update rule. The function to be optimized is input here.
   *
   * @param function Function to be optimized in FrankWolfe algorithm.
   */
  UpdateSpan(const bool isPrune = false) : isPrune(isPrune)
  { /* Do nothing. */ }

  /**
   * Update rule for FrankWolfe, reoptimize in the span of current
   * solution space.
   *
   * @param function function to be optimized.
   * @param oldCoords previous solution coords.
   * @param s current linearConstrSolution result.
   * @param newCoords output new solution coords.
   * @param numIter current iteration number.
   */
  void Update(FuncSq& function,
              const arma::mat& oldCoords,
              const arma::mat& s,
              arma::mat& newCoords,
              const size_t /* numIter */)
  {
    // Add new atom into soluton space.
    atoms.AddAtom(s, function);

    // Reoptimize the solution in the current space.
    arma::vec b = function.Vectorb();
    atoms.CurrentCoeffs() = solve(function.MatrixA() * atoms.CurrentAtoms(), b);

    // x has coords of only the current atoms, recover the solution
    // to the original size.
    atoms.RecoverVector(newCoords);

    // Prune the support.
    if (isPrune)
    {
      double oldF = function.Evaluate(oldCoords);
      double F = 0.25 * oldF + 0.75 * function.Evaluate(newCoords);
      atoms.PruneSupport(F, function);
      atoms.RecoverVector(newCoords);
    }
  }

 private:
  //! Atoms information.
  Atoms atoms;

  //! Flag for support prune step.
  bool isPrune;
}; // class UpdateSpan

} // namespace optimization
} // namespace mlpack

#endif
