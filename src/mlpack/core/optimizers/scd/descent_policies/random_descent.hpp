/**
 * @file random_descent.hpp
 * @author Shikhar Bhardwaj
 *
 * Random descent policy for Stochastic Coordinate Descent (SCD).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_SCD_DESCENT_POLICIES_RANDOM_HPP
#define MLPACK_CORE_OPTIMIZERS_SCD_DESCENT_POLICIES_RANDOM_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace optimization {

/**
 * Random descent policy for Stochastic Coordinate Descent(SCD). This
 * descent scheme picks a the co-ordinate for the descent uniformly randomly.
 *
 * For more information, see the following.
 * @code
 * @inproceedings{Shalev-Shwartz:2009:SML:1553374.1553493,
 *   Author = {Shalev-Shwartz, Shai and Tewari, Ambuj},
 *   Title = {Stochastic Methods for L1 Regularized Loss Minimization},
 *   Booktitle = {Proceedings of the 26th Annual International Conference on Machine Learning},
 *   Series = {ICML '09},
 *   Year = {2009},
 *   isbn = {978-1-60558-516-1},
 *   url = {http://doi.acm.org/10.1145/1553374.1553493},
 * }
 * @endcode
 */
class RandomDescent
{
 public:
  /**
   * The DescentFeature method is used to get the descent coordinate for the
   * current iteration of the SCD optimizer. For more information regarding the
   * interface of this policy with the optimizer, have a look at the SCD
   * implementation.
   *
   * @tparam ResolvableFunctionType The type of the function to be optimized.
   * @param iteration The iteration number for which the feature is to be
   *    obtained.
   * @param iterate The current value of the decision variable.
   * @param function The function to be optimized.
   * @return The index of the coordinate to be descended.
   */
  template <typename ResolvableFunctionType>
  static size_t DescentFeature(const size_t /* iteration */,
                        const arma::mat& /* iterate */,
                        const ResolvableFunctionType& function)
  {
    return mlpack::math::RandInt(function.NumFeatures());
  }
};

} // namespace optimization
} // namespace mlpack

#endif
