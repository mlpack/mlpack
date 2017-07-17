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

namespace mlpack {
namespace optimization {

/**
 * Recalculate the optimal solution in the span of all previous solution space,
 * used as update step for FrankWolfe algorithm.
 *
 * Currently only works for function in FuncSq class.
 *
 */
class UpdateSpan
{
 public:
  /**
   * Construct the span update rule. The function to be optimized is 
   * input here.
   *
   * @param function Function to be optimized in FrankWolfe algorithm.
   */
  UpdateSpan()
  { /* Do nothing. */ }

  /**
   * Update rule for FrankWolfe, reoptimize in the span of current
   * solution space.
   *
   * @param function function to be optimized.
   * @param oldCoords previous solution coords, not used in this update rule.
   * @param s current linearConstrSolution result.
   * @param newCoords output new solution coords.
   * @param numIter current iteration number.
   */
  void Update(FuncSq& function,
              const arma::mat& oldCoords,
              const arma::mat& s,
              arma::mat& newCoords,
              const size_t numIter)
  {
    // Add new atom into soluton space.
    arma::uvec ind = find(s, 1);
    AddAtom(function, ind(0));

    // Reoptimize the solution in the current space.
    arma::vec b = function.Vectorb();
    arma::vec x = solve(currentAtoms, b);

    // x has coords of only the current atoms, recover the solution
    // to the original size.
    RecoverVector(x, function.MatrixA().n_cols, newCoords);
  }

  //! Get the current atom indices.
  const arma::uvec& CurrentIndices() const { return currentIndices; }
  //! Modify the current atom indices.
  arma::uvec& CurrentIndices() { return currentIndices; }

  //! Get the current atoms.
  const arma::mat& CurrentAtoms() const { return currentAtoms; }
  //! Modify the current atoms.
  arma::mat& CurrentAtoms() { return currentAtoms; }

 private:
  //! Indices of the atoms in the current solution space.
  arma::uvec currentIndices;

  //! Current atoms in the solution space, ordered as currentIndices.
  arma::mat currentAtoms;

  //! Add atom into the solution space.
  void AddAtom(FuncSq& function, const arma::uword k)
  {
    if (currentIndices.is_empty())
    {
      CurrentIndices() = k;
      CurrentAtoms() = (function.MatrixA()).col(k);
    }
    else
    {
      arma::uvec vk(1);
      vk = k;
      currentIndices.insert_rows(0, vk);

      arma::mat atom = (function.MatrixA()).col(k);
      currentAtoms.insert_cols(0, atom);
    }
  }

  /**
   * Recover the solution coordinate from the coefficients of current atoms.
   *
   * @param x input coefficients of current atoms.
   * @param n dimension of the recovered vector.
   * @param y output recovered vector.
   */
  void RecoverVector(const arma::vec& x, const size_t n, arma::mat& y)
  {
    y.zeros(n, 1);
    arma::uword len = currentIndices.size();
    for (size_t ii = 0; ii < len; ++ii)
      y(currentIndices(ii)) = x(ii);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
