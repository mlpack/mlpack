/**
 * @file ip_metric.hpp
 * @author Ryan Curtin
 *
 * Inner product induced metric.  If given a kernel function, this gives the
 * complementary metric.
 */
#ifndef __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP
#define __MLPACK_METHODS_FASTMKS_IP_METRIC_HPP

namespace mlpack {
namespace fastmks /** Fast maximum kernel search. */ {

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

 private:
  //! The locally stored kernel, if it is necessary.
  KernelType* localKernel;
  //! The reference to the kernel that is being used.
  KernelType& kernel;
};

}; // namespace fastmks
}; // namespace mlpack

// Include implementation.
#include "ip_metric_impl.hpp"

#endif
