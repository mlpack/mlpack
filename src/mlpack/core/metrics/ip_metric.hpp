/**
 * @file ip_metric.hpp
 * @author Ryan Curtin
 *
 * Inner product induced metric.  If given a kernel function, this gives the
 * complementary metric.
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
#ifndef __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP
#define __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP

namespace mlpack {
namespace metric {

template<typename KernelType>
class IPMetric
{
 public:
  //! Create the IPMetric without an instantiated kernel.
  IPMetric();

  //! Create the IPMetric with an instantiated kernel.
  IPMetric(KernelType& kernel);

  //! Destroy the IPMetric object.
  ~IPMetric();

  /**
   * Evaluate the metric.
   */
  template<typename Vec1Type, typename Vec2Type>
  double Evaluate(const Vec1Type& a, const Vec2Type& b);

  //! Get the kernel.
  const KernelType& Kernel() const { return kernel; }
  //! Modify the kernel.
  KernelType& Kernel() { return kernel; }
  /**
   * Returns a string representation of this object.
   */
  std::string ToString() const;
 private:
  //! The locally stored kernel, if it is necessary.
  KernelType* localKernel;
  //! The reference to the kernel that is being used.
  KernelType& kernel;
};

}; // namespace metric
}; // namespace mlpack

// Include implementation.
#include "ip_metric_impl.hpp"

#endif
