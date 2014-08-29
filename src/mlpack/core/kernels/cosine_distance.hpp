/**
 * @file cosine_distance.hpp
 * @author Ryan Curtin
 *
 * This implements the cosine distance (or cosine similarity) between two
 * vectors, which is a measure of the angle between the two vectors.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP
#define __MLPACK_CORE_KERNELS_COSINE_DISTANCE_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * The cosine distance (or cosine similarity).  It is defined by
 *
 * @f[
 * d(a, b) = \frac{a^T b}{|| a || || b ||}
 * @f]
 *
 * and this class assumes the standard L2 inner product.
 */
class CosineDistance
{
 public:
  /**
   * Computes the cosine distance between two points.
   *
   * @param a First vector.
   * @param b Second vector.
   * @return d(a, b).
   */
  template<typename VecType>
  static double Evaluate(const VecType& a, const VecType& b);

  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "CosineDistance [" << this << "]" << std::endl;
    return convert.str();
  }
};

//! Kernel traits for the cosine distance.
template<>
class KernelTraits<CosineDistance>
{
 public:
  //! The cosine kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
};

}; // namespace kernel
}; // namespace mlpack

// Include implementation.
#include "cosine_distance_impl.hpp"

#endif
