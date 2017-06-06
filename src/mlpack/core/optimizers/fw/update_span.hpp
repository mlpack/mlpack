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
 *
 *
 * @tparam FunctionType Objective function type to be minimized in FrankWolfe algorithm.
 */
template<typename FunctionType>
class UpdateSpan
{
 public:
  /**
   * Construct the span update rule. The function to be optimized is 
   * input here.
   *
   * @param function Function to be optimized in FrankWolfe algorithm.
   */
  UpdateSpan(FunctionType& function): function(function)    
  { /* Do nothing. */ }
  // Chenzhe: should we use reference to function? what's the difference? 

 /**
  * Update rule for FrankWolfe, reoptimize in the span of original solution space.
  *
  *
  * @param old_coords previous solution coords.
  * @param s current linear_constr_solution result.
  * @param new_coords new output solution coords.
  * @param num_iter current iteration number
  */
  void Update(const arma::mat& old_coords,
	  const arma::mat& s,
	  arma::mat& new_coords,
	  const size_t num_iter)
  {
      //Need to add atom here
      arma::uvec ind = find(s, 1);
      arma::uword d = ind(0);
      AddAtom(d);

      arma::vec b = function.Vectorb();
      arma::mat x = solve(atoms_current, b);

      new_coords = RecoverVector(x);
  }



  //! Get the instantiated function to be optimized.
  FunctionType Function() const { return function; }
  //! Modify the instantiated function.
  FunctionType& Function() { return function; }

  //! Get the current atom indices.
  arma::uvec CurrentIndices() const { return current_indices; }
  //! Modify the current atom indices.
  arma::uvec& CurrentIndices() { return current_indices; }

  //! Get the current atoms.
  arma::mat CurrentAtoms() const { return atoms_current; }
  //! Modify the current atoms.
  arma::mat& CurrentAtoms() { return atoms_current; }

  //! Add atom into the solution space, modify the current_indices and atoms_current.
  void AddAtom(const arma::uword k) 
  {
      if (isEmpty){
	  CurrentIndices() = k;
	  CurrentAtoms() = (function.MatrixA()).col(k);
	  isEmpty = false;
      }
      else{
	  arma::uvec vk(1);
	  vk = k;
	  current_indices.insert_rows(0, vk);  

	  arma::mat atom = (function.MatrixA()).col(k);
	  atoms_current.insert_cols(0, atom);
	  
      }
  }

  arma::vec RecoverVector(const arma::vec& x )
  {
      int n = (function.MatrixA()).n_cols;
      arma::vec y = arma::zeros<arma::vec>(n);

      arma::uword len = current_indices.size();
      for (size_t ii = 0; ii < len; ++ii)
      {
	  y(current_indices(ii)) = x(ii);
      }

      return y;
  }

 private:
  //! The instantiated function.
  FunctionType& function;

  //! Current indices.
  arma::uvec current_indices;		

  //! Current atoms.
  arma::mat atoms_current;

  //! Flag current indices is empty
  bool isEmpty=true;

};

} // namespace optimization
} // namespace mlpack

#endif
