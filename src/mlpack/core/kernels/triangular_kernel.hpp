/**
 * @file triangular_kernel.hpp
 * @author Ryan Curtin
 *
 * Definition and implementation of the trivially simple triangular kernel.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

/**
 * The trivially simple triangular kernel, defined by
 *
 * @f[
 * K(x, y) = \max \{ 0, 1 - \frac{|| x - y ||_2}{b} \}
 * @f]
 *
 * where \f$ b \f$ is the bandwidth of the kernel.
 */
class TriangularKernel
{
 public:
  /**
   * Initialize the triangular kernel with the given bandwidth (default 1.0).
   *
   * @param bandwidth Bandwidth of the triangular kernel.
   */
  TriangularKernel(const double bandwidth = 1.0) : bandwidth(bandwidth) { }

  /**
   * Evaluate the triangular kernel for the two given vectors.
   *
   * @param a First vector.
   * @param b Second vector.
   */
  template<typename Vec1Type, typename Vec2Type>
  double Evaluate(const Vec1Type& a, const Vec2Type& b) const
  {
    return std::max(0.0, (1 - metric::EuclideanDistance::Evaluate(a, b) /
        bandwidth));
  }

  /**
   * Evaluate the triangular kernel given that the distance between the two
   * points is known.
   *
   * @param distance The distance between the two points.
   */
  double Evaluate(const double distance) const
  {
    return std::max(0.0, (1 - distance) / bandwidth);
  }

  //! Get the bandwidth of the kernel.
  double Bandwidth() const { return bandwidth; }
  //! Modify the bandwidth of the kernel.
  double& Bandwidth() { return bandwidth; }

  //! Return a string representation of the kernel.
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "TriangularKernel [" << this << "]" << std::endl;
    convert << "  Bandwidth: " << bandwidth << std::endl;
    return convert.str();
  }

 private:
  //! The bandwidth of the kernel.
  double bandwidth;
};

//! Kernel traits for the triangular kernel.
template<>
class KernelTraits<TriangularKernel>
{
 public:
  //! The triangular kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
};

}; // namespace kernel
}; // namespace mlpack

#endif
