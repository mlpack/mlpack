/**
 * @file average_interpolation.hpp
 * @author Wenhao Huang
 *
 * This class performs average interpolation to generate interpolation weights
 * for neighborhood-based collaborative filtering.
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
  * @param weights resulting interpolation weights.
  * @param distances distances from NeighborSearch::Search().
  */
  void GetWeights(arma::vec& weights, const arma::vec& distances) const
  {
    weights.resize(distances.n_elem);
    if (distances.n_elem == 0)
    {
      Log::Fatal << "Require: distances.n_elem > 0. There should be at least "
          << "one neighbor!"
          << std::endl;
    }
    weights.fill(1.0 / distances.n_elem);
  }
};

} // namespace cf
} // namespace mlpack

#endif
