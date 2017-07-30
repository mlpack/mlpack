/**
 * @file scd.hpp
 * @author Shikhar Bhardwaj
 *
 * Stochastic Co ordinate Descent (SCD).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SCD_SCD_HPP
#define MLPACK_CORE_OPTIMIZERS_SCD_SCD_HPP

#include <mlpack/prereqs.hpp>
#include "descent_policies/random_descent.hpp"

namespace mlpack {
namespace optimization {

/**
 * Stochastic Co ordinate descent is a technique for minimizing a function by
 * doing a line search along a single direction at the current point in the
 * iteration. The direction (or "coordinate") can be chosen cyclically, randomly
 * or in a greedy fashion(depending on the DescentPolicy).
 *
 * This optimizer is useful for problems with a smooth multivariate function
 * where computing the entire gradient for an update is infeasable. CD method
 * typically significantly outperform GD, especially on sparse problems with a
 * very large number variables/coordinates.
 *
 * For SCD to work, a ResolvableFunctionType template parameter is required.
 * This Class must implement the following functions:
 *
 *  size_t NumFeatures();
 *  double Evaluate(const arma::mat& coordinates);
 *  void FeatureGradient(const arma::mat& coordinates,
 *                       const size_t j,
 *                       double& gradient);
 *
 *  NumFeatures() should return the number of features in the decision variable.
 *  Evaluate gives the value of the loss function at the current decision
 *  variable and FeatureGradient is used to evaluate the partial gradient with
 *  respect to the jth feature.
 *
 *  @tparam ResolvableFunctionType A function whose partial gradients with
 *      respect to the jth feature can be obtained.
 *  @tparam DescentPolicy Descent policy to decide the order in which the
 *      coordinate for descent is selected.
 */
template <typename DescentPolicyType = RandomDescent>
class SCD
{
  public:
};

} 
} // namespace mlpack
#endif
