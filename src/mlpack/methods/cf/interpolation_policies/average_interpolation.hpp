/**
 * @file average_interpolation.hpp
 * @author Wenhao Huang
 *
 * Interoplation weights are identical and sum up to one.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_METHODS_CF_AVERAGE_INTERPOLATION_HPP
#define MLPACK_METHODS_CF_AVERAGE_INTERPOLATION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace cf {

/**
 * This class performs average interpolation to generate interpolation weights
 * for neighborhood-based collaborative filtering.
 */
class AverageInterpolation
{
 public:
  // Empty constructor.
  AverageInterpolation() { }

  /**
   * Interoplation weights are identical and sum up to one.
   *
   * @param weights Resulting interpolation weights.
   * @param similarities Similarites between query user and neighbors.
   */
  void GetWeights(arma::vec& weights, const arma::vec& similarities) const
  {
    if (similarities.n_elem == 0)
    {
      Log::Fatal << "Require: similarities.n_elem > 0. There should be at "
          << "least one neighbor!" << std::endl;
    }
    weights.set_size(similarities.n_elem);
    weights.fill(1.0 / similarities.n_elem);
  }
};

} // namespace cf
} // namespace mlpack

#endif
