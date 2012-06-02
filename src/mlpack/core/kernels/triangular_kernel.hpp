/**
 * @file triangular_kernel.hpp
 * @author Ryan Curtin
 *
 * Triangular kernel.  Is it a Mercer kernel?  Who knows!?
 */
#ifndef __MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_TRIANGULAR_KERNEL_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

class TriangularKernel
{
 public:
  template<typename VecType>
  static double Evaluate(const VecType& a, const VecType& b)
  {
    return std::max(0.0, (1 - metric::LMetric<2>::Evaluate(a, b)));
  }
};

}; // namespace kernel
}; // namespace mlpack

#endif
