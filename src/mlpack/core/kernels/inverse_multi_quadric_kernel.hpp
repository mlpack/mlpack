/**
 * @file inverse_multi_quadric_kernel.hpp
 * @author Ryan Curtin
 *
 * The deadline is coming!
 */
#ifndef __MLPACK_CORE_KERNELS_INVERSE_MULTI_QUADRIC_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_INVERSE_MULTI_QUADRIC_KERNEL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

class InverseMultiQuadricKernel
{
 public:
  template<typename VecType>
  static double Evaluate(const VecType& a, const VecType& b)
  {
    return 1.0 / (metric::EuclideanDistance::Evaluate(a, b));
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
