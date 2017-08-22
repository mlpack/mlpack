/**
 * @file mc_fw_update_matrix.hpp
 * @author Chenzhe Diao
 *
 * In FrankWolfe algorithm, used as UpdateRuleType.
 * Designed for matrix completion problems.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_UPDATE_MATRIX_HPP
#define MLPACK_METHODS_MATRIX_COMPLETION_MC_FW_UPDATE_MATRIX_HPP

#include "mc_fw_function.hpp"

namespace mlpack {
namespace matrix_completion {

class UpdateMatrix {
 public:
  UpdateMatrix(double tau): tau(tau) {/* Nothing to do. */}

  void Update(MatrixCompletionFWFunction& function,
              const arma::mat& oldCoords,
              const arma::mat& s,
              arma::mat& newCoords,
              const size_t numIter)
  {
    // Line search, with explicit solution here.
    arma::mat v = tau * s - oldCoords;
    arma::vec b = function.Values();
    double gamma = arma::dot(b - function.GetKnownEntries(oldCoords),
        function.GetKnownEntries(v));
    gamma = gamma / std::pow(arma::norm(function.GetKnownEntries(v), "fro"), 2);
    gamma = std::min(gamma, 1.0);
    currentCoeffs = (1.0 - gamma) * currentCoeffs;
    AddAtom(s, gamma * tau);

    RecoverVector(newCoords);
  }


 private:
  //! Atom norm constraint.
  double tau;

  //! Current matrix Atoms.
  arma::cube currentAtoms;

  //! Current coefficients of the atoms.
  arma::vec currentCoeffs;


  void AddAtom(const arma::mat& v, const double c = 0)
  {
    if (currentAtoms.is_empty())
    {
      currentAtoms = v;
      currentCoeffs.set_size(1);
      currentCoeffs.fill(c);
    }
    else
    {
      currentAtoms.insert_slices(0, v);
      arma::vec cVec(1);
      cVec(0) = c;
      currentCoeffs.insert_rows(0, cVec);
    }
  }

  //! Recover the solution coordinate from the coefficients of current atoms.
  void RecoverVector(arma::mat& x)
  {
    x = arma::zeros<arma::mat>(currentAtoms.n_rows, currentAtoms.n_cols);
    for (i = 0; i < currentCoeffs.n_rows; i++)
      x += currentAtoms.slice(i) * currentCoeffs(i);
  }


};  // class UpdateMatrix



} // namespace matrix_completion
} // namespace mlpack

#endif
