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

namespace mlpack {
namespace optimization {

/**
 * Recalculate the optimal solution in the span of all previous solution space, 
 * used as update step for FrankWolfe algorithm.
 *
 *
 * For UpdateSpan to work, FunctionType template parameters are required.
 * This class must implement the following functions:
 *
 * FunctionType:
 *
 *   double Evaluate(const arma::mat& coordinates);
 *   void Gradient(const arma::mat& coordinates,
 *                 arma::mat& gradient);
 *   arma::mat MatrixA()
 *   arma::vec Vectorb()
 *
 *
 * @tparam FunctionType Objective function type to be minimized in 
 *                      FrankWolfe algorithm.
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
   * Update rule for FrankWolfe, reoptimize in the span of original 
   * solution space.
   *
   * @tparam function function to be optimized.
   * @param oldCoords previous solution coords, not used in this update rule.
   * @param s current linearConstrSolution result.
   * @param newCoords new output solution coords.
   * @param numIter current iteration number.
   */
  template<typename FunctionType>
  void Update(FunctionType& function,
              const arma::mat& oldCoords,
              const arma::mat& s,
              arma::mat& newCoords,
              const size_t numIter)
  {
    // Add atom here.
    arma::uvec ind = find(s, 1);
    arma::uword d = ind(0);
    AddAtom(function, d);

    arma::vec b = function.Vectorb();
    arma::mat x = solve(currentAtoms, b);

    newCoords = RecoverVector(function, x);
  }

  //! Get the current atom indices.
  arma::uvec CurrentIndices() const { return currentIndices; }
  //! Modify the current atom indices.
  arma::uvec& CurrentIndices() { return currentIndices; }

  //! Get the current atoms.
  arma::mat CurrentAtoms() const { return currentAtoms; }
  //! Modify the current atoms.
  arma::mat& CurrentAtoms() { return currentAtoms; }

 private:
  //! Current indices.
  arma::uvec currentIndices;

  //! Current atoms.
  arma::mat currentAtoms;

  //! Flag current indices is empty.
  bool isEmpty = true;

  //! Add atom into the solution space.
  template<typename FunctionType>
  void AddAtom(FunctionType& function, const arma::uword k)
  {
    if (isEmpty)
    {
      CurrentIndices() = k;
      CurrentAtoms() = (function.MatrixA()).col(k);
      isEmpty = false;
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

  template<typename FunctionType>
  arma::vec RecoverVector(FunctionType& function, const arma::vec& x)
  {
    int n = (function.MatrixA()).n_cols;
    arma::vec y = arma::zeros<arma::vec>(n);

    arma::uword len = currentIndices.size();
    for (size_t ii = 0; ii < len; ++ii)
      y(currentIndices(ii)) = x(ii);

    return y;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
