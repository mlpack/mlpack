/***
 * @file cosine_distance.hpp
 * @author Tommi Laivamaa
 *
 * The cosine distance.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_METRICS_COSINE_DISTANCE_HPP
#define MLPACK_CORE_METRICS_COSINE_DISTANCE_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace metric {

/**
 * The cosine distance, which is defined as
 *
 * @f[
 * d(x, y) = 1 - \frac{x \cdot y}{||x|| \cdot ||y||}.
 * @f]
 *
 * Note that the cosine distance is not a proper metric as the triangle
 * inequality does not hold in all cases.
 */
class CosineDistance
{
 public:
  /**
   * Initialize the cosine distance.
   */
  CosineDistance() { }

  /**
   * Evaluate the distance between the two given points using the cosine
   * distance.
   *
   * @tparam VecTypeA Type of the first vector (arma::vec, arma::spvec).
   * @tparam VecTypeB Type of the second vector (arma::vec, arma::spvec).
   * @param a First vector.
   * @param b Second vector.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b)
  {
    return 1 - arma::dot(a, b) / (std::sqrt(arma::dot(a, a) * arma::dot(b, b)));
  }
};

} // namespace metric
} // namespace mlpack

#endif
