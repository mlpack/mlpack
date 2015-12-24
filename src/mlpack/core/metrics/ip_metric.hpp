/**
 * @file ip_metric.hpp
 * @author Ryan Curtin
 *
 * Inner product induced metric.  If given a kernel function, this gives the
 * complementary metric.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP
#define __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP

namespace mlpack {
namespace metric {

/**
 * The inner product metric, IPMetric, takes a given Mercer kernel (KernelType),
 * and when Evaluate() is called, returns the distance between the two points in
 * kernel space:
 *
 * @f[
 * d(x, y) = \sqrt{ K(x, x) + K(y, y) - 2K(x, y) }.
 * @f]
 *
 * @tparam KernelType Type of Kernel to use.  This must be a Mercer kernel
 *     (positive definite), otherwise the metric may not be valid.
 */
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
   *
   * @tparam VecTypeA Type of first vector.
   * @tparam VecTypeB Type of second vector.
   * @param a First vector.
   * @param b Second vector.
   * @return Distance between the two points in kernel space.
   */
  template<typename VecTypeA, typename VecTypeB>
  double Evaluate(const VecTypeA& a, const VecTypeB& b);

  //! Get the kernel.
  const KernelType& Kernel() const { return *kernel; }
  //! Modify the kernel.
  KernelType& Kernel() { return *kernel; }

  //! Serialize the metric.
  template<typename Archive>
  void Serialize(Archive& ar, const unsigned int version);

 private:
  //! The kernel we are using.
  KernelType* kernel;
  //! If true, we are responsible for deleting the kernel.
  bool kernelOwner;
};

} // namespace metric
} // namespace mlpack

// Include implementation.
#include "ip_metric_impl.hpp"

#endif
