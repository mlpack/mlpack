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

    arma::vec oldEntries, vEntries;
    function.GetKnownEntries(oldCoords, oldEntries);
    function.GetKnownEntries(v, vEntries);

    double gamma = arma::dot(b - oldEntries, vEntries);
    gamma = gamma / std::pow(arma::norm(vEntries, "fro"), 2);
    gamma = std::min(gamma, 1.0);
    currentCoeffs = (1.0 - gamma) * currentCoeffs;
    AddAtom(s, gamma * tau);

    RecoverVector(newCoords);
    double fOld = function.Evaluate(oldCoords);
    double fNew = function.Evaluate(newCoords);
    double f = 0.5 * fOld + 0.5 * fNew;
    Truncation(f, function);

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
      currentAtoms.set_size(v.n_rows, v.n_cols, 1);
      currentAtoms.slice(0) = v;
      currentCoeffs.set_size(1);
      currentCoeffs.fill(c);
    }
    else
    {
      arma::cube vCube(v.n_rows, v.n_cols, 1);
      vCube.slice(0) = v;
      currentAtoms.insert_slices(0, vCube);
      arma::vec cVec(1);
      cVec(0) = c;
      currentCoeffs.insert_rows(0, cVec);
    }
  }

  //! Recover the solution coordinate from the coefficients of current atoms.
  void RecoverVector(arma::mat& x)
  {
    x = arma::zeros<arma::mat>(currentAtoms.n_rows, currentAtoms.n_cols);
    for (arma::uword i = 0; i < currentCoeffs.n_rows; i++)
      x += currentAtoms.slice(i) * currentCoeffs(i);
  }
  
  void Truncation(const double F, MatrixCompletionFWFunction& function)
  {
    arma::mat X;
    RecoverVector(X);

    arma::mat U, V;
    arma::vec s;
    arma::uword k;
    if(!arma::svd_econ(U, s, V, X))
      Log::Fatal << "Truncation: armadillo svd_econ() failed!";
    size_t rank = s.n_elem;
    
    for (size_t i = 1; i < rank; i++)
    {
      // Try deleting the rank one matrix with smallest coefficient.
      s.min(k);
      X = X - s(k) * U.col(k) * arma::trans(V.col(k));
      
      // Cannot delete atom, just break.
      if (function.Evaluate(X) > F)
        break;
      
      // Delete this atom.
      U.shed_col(k);
      V.shed_col(k);
      s.shed_row(k);
    } // for

    // Redefine the atoms.
    currentAtoms.set_size(U.n_rows, V.n_rows, s.n_rows);
    currentCoeffs = s;
    for (arma::uword i = 0; i < s.n_rows; i++)
      currentAtoms.slice(i) = U.col(i) * arma::trans(V.col(i));
  } // Truncation()

};  // class UpdateMatrix



} // namespace matrix_completion
} // namespace mlpack

#endif
