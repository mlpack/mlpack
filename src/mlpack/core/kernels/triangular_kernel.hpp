/**
 * @file triangular_kernel.hpp
 * @author Ryan Curtin
 *
 * Definition and implementation of the trivially simple triangular kernel.
 * This file is part of MLPACK 1.0.2.
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
  double Evaluate(const Vec1Type& a, const Vec2Type& b)
  {
    return std::max(0.0, (1 - metric::EuclideanDistance::Evaluate(a, b) /
        bandwidth));
  }

  //! Get the bandwidth of the kernel.
  double Bandwidth() const { return bandwidth; }
  //! Modify the bandwidth of the kernel.
  double& Bandwidth() { return bandwidth; }

 private:
  //! The bandwidth of the kernel.
  double bandwidth;
};

}; // namespace kernel
}; // namespace mlpack

#endif
