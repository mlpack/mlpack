/**
 * @file frank_wolfe_impl.hpp
 * @author Chenzhe Diao
 *
 * Frank-Wolfe Algorithm.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FW_FRANK_WOLFE_IMPL_HPP
#define MLPACK_CORE_OPTIMIZERS_FW_FRANK_WOLFE_IMPL_HPP

// In case it hasn't been included yet.
#include "frank_wolfe.hpp"

namespace mlpack {
namespace optimization {

template<
    typename FunctionType, 
    typename LinearConstrSolverType, 
    typename UpdateRuleType
>
FrankWolfe<FunctionType, LinearConstrSolverType, UpdateRuleType>::FrankWolfe(
    FunctionType& function,
    const LinearConstrSolverType linear_constr_solver,
    const UpdateRuleType update_rule,
    const size_t maxIterations,
    const double tolerance) :
    function(function),
    linear_constr_solver(linear_constr_solver),
    update_rule(update_rule),
    maxIterations(maxIterations),
    tolerance(tolerance)
{ /* Nothing to do*/ }


//! Optimize the function (minimize).  
template<
    typename FunctionType, 
    typename LinearConstrSolverType, 
    typename UpdateRuleType
>
double FrankWolfe<FunctionType, LinearConstrSolverType, UpdateRuleType>::Optimize(
    arma::mat& iterate)
{
    // To keep track of the function value
    double CurrentObjective = function.Evaluate(iterate);
    double PreviousObjective = DBL_MAX;

    arma::mat gradient(iterate.n_rows, iterate.n_cols);
    arma::mat s(iterate.n_rows, iterate.n_cols);
    arma::mat iterate_new(iterate.n_rows, iterate.n_cols);
    double gap = 0;

    for(size_t i=1; i != maxIterations; ++i)
    {
	//Output current objective function
	Log::Info << "Iteration " << i << ", objective "
	    << CurrentObjective << "." << std::endl;

	// Reset counter variables.
	PreviousObjective = CurrentObjective;
	
	// Calculate the gradient
	function.Gradient(iterate, gradient);

	// Solve linear constrained problem, solution saved in s.
	linear_constr_solver.Optimize(gradient, s);

	// Check duality gap for return condition
	gap = dot(iterate-s, gradient);
	if (gap < tolerance)
	{
	    Log::Info << "FrankWolfe: minimized within tolerance "
		<< tolerance << "; " << "terminating optimization." << std::endl;
	    return CurrentObjective;
	}


	// Update solution, save in iterate_new
	update_rule.Update(iterate, s, iterate_new, i);

	
	iterate = iterate_new;
	CurrentObjective = function.Evaluate(iterate);
    }
  Log::Info << "Frank Wolfe: maximum iterations (" << maxIterations
      << ") reached; " << "terminating optimization." << std::endl;
  return CurrentObjective;

}


} // namespace optimization
} // namespace mlpack

#endif
