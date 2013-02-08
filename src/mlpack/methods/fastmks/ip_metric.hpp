/**
 * @file ip_metric.hpp
 * @author Ryan Curtin
 *
 * Inner product induced metric.  If given a kernel function, this gives the
 * complementary metric.
 *
 * This file is part of MLPACK 1.0.4.
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
#ifndef __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP
#define __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP

namespace mlpack {
namespace fastmks /** The fast maximum kernel search problem. */ {

template<typename KernelType>
class IPMetric
{
 public:
  IPMetric() : kernel(localKernel) { }

  /**
   * Create the IPMetric.
   */
  IPMetric(KernelType& kernel) : kernel(kernel) { }

  /**
   * Evaluate the metric.
   */
  template<typename Vec1Type, typename Vec2Type>
  double Evaluate(const Vec1Type& a, const Vec2Type& b);

  const KernelType& Kernel() const { return kernel; }
  KernelType& Kernel() { return kernel; }

  KernelType localKernel;
  KernelType& kernel;
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "ip_metric_impl.hpp"

#endif
