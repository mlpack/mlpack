/**
 * @file atoms.hpp
 * @author Chenzhe Diao
 *
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_ATOMS_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_ATOMS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

class Atoms
{
 public:
  Atoms(){}
  
  //! Add atom into the solution space.
  void AddAtom(const arma::mat& v, const double c = 0)
  {
    if (currentAtoms.is_empty())
    {
      CurrentAtoms() = v;
      CurrentCoeffs() = c;
    }
    else
    {
      currentAtoms.insert_cols(0, v);
      currentCoeffs.insert_rows(0, c);
    }
  }
  
  /**
   * Recover the solution coordinate from the coefficients of current atoms.
   *
   * @param y output recovered vector.
   */
  void RecoverVector(arma::mat& x)
  {
    x = currentAtoms * currentCoeffs;
  }

//  template<typename FunctionType>
//  void ProjectedGradientEnhancement(double tau,
//                                    FunctionType& function,
//                                    double stepSize,
//                                    size_t maxIteration = 100,
//                                    double tolerance = 1e-3)
//  {
//    // Gradient Descent.
//    arma::mat g;
//    arma::mat x;
//    RecoverVector(x);
//    double value = function.Evaluate(x);
//    
//    for (size_t iter = 1; iter<maxIteration; iter++)
//    {
//      function.Gradient(x, g);
//      g = currentAtoms.t() * g;
//      currentCoeffs = currentCoeffs - stepSize * g;
//
//      // Projection
//      ProjectionToL1(tau);
//
//      RecoverVector(x);
//      double valueNew = function.Evaluate(x);
//
//      if (std::abs(value - valueNew) < tolerance)
//        break;
//
//      value = valueNew;
//    }
//    
//    
//  }
  
//  //! Prune the support, delete previous atoms if not necessary.
//  void PruneSupport(const double F)
//  {
//    arma::mat newAtoms = currentAtoms;
//    arma::vec newCoeff = currentCoeffs;
//    arma::vec b = function.Vectorb();
//    
//    bool flag = true;
//    
//    while (flag)
//    {
//      // Solve for current gradient
//      arma::vec g;
//      function.GradientFunc(new_coeff, g, new_atoms, b);
//      
//      // Find possible atom to be deleted
//      arma::vec v = sum(new_atoms % new_atoms, 0);
//      v = 0.5*v.t() % new_coeff % new_coeff - new_coeff % g;
//      arma::uword ind = v.index_min();
//      
//      // Try deleting the atom.
//      new_atoms.shed_row(ind);
//      new_indices.shed_row(ind);
//      new_coeff = solve(new_atoms, b);  // recalculate the coeff
//      double F_new = function.EvaluateFunc(new_coeff, new_atoms, b);
//      
//      if (F_new > F)
//        // should not delete the atom
//        flag = false;
//      else {
//        // delete the atom from current atoms
//        atoms_current = new_atoms;
//        current_indices = new_indices;
//        x = new_coeff;
//      }
//    }
//  }

  
  //! Get the current atom coefficients.
  const arma::vec& CurrentCoeffs() const { return currentCoeffs; }
  //! Modify the current atom coefficients.
  arma::vec& CurrentCoeffs() { return currentCoeffs; }
  
  //! Get the current atoms.
  const arma::mat& CurrentAtoms() const { return currentAtoms; }
  //! Modify the current atoms.
  arma::mat& CurrentAtoms() { return currentAtoms; }
  
 private:
  //! Coefficients of current atoms.
  arma::vec currentCoeffs;
  
  //! Current atoms in the solution space.
  arma::mat currentAtoms;
  
  //! Flag for support prune.
  bool isPrune = false;

  // Projection to L1 ball with norm tau.
//  void ProjectionToL1(const double tau)
//  {
//    arma::vec tmp = arma::abs(currentCoeffs);
//    if (arma::accu(tmp) <= tau)
//      return;
//
//    arma::uvec ind = arma::sort_index(tmp, "descend");
//    tmp = tmp(ind);
//    arma::vec tmpSum = arma::cumsum(tmp);
//
//    double nu = 0;
//    size_t rho;
//    for (rho = tmp.n_rows-1; rho >= 0; rho--)
//    {
//      nu = tmp(rho) - (tmpSum(rho) - tau)/rho;
//      if (nu <= 0)
//      {
//        rho++;
//        break;
//      }
//    }
//    double theta = (tmpSum(rho) - tau)/rho;
//    
//    // Threshold on currentCoeffs with theta.
//    for (arma::uword j = 0; j< tmp.n_rows; j++)
//    {
//      if (currentCoeffs(j) >=0)
//        currentCoeffs(j) = std::max(currentCoeffs(j)-theta, 0);
//      else
//        currentcoeffs(j) = std::min(currentCoeffs(j)+theta, 0);
//    }
//    
//  }
  

}; // class Atoms
}  // namespace optimization
}  // namespace mlpack


#endif
