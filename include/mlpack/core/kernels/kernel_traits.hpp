/**
 * @file core/kernels/kernel_traits.hpp
 * @author Ryan Curtin
 *
 * This provides the KernelTraits class, a template class to get information
 * about various kernels.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_KERNELS_KERNEL_TRAITS_HPP
#define MLPACK_CORE_KERNELS_KERNEL_TRAITS_HPP

namespace mlpack {

/**
 * This is a template class that can provide information about various kernels.
 * By default, this class will provide the weakest possible assumptions on
 * kernels, and each kernel should override values as necessary.  If a kernel
 * doesn't need to override a value, then there's no need to write a
 * KernelTraits specialization for that class.
 */
template<typename KernelType>
class KernelTraits
{
 public:
  /**
   * If true, then the kernel is normalized: K(x, x) = K(y, y) = 1 for all x.
   */
  static const bool IsNormalized = false;

  /**
   * If true, then the kernel include a squared distance, ||x - y||^2 .
   */
  static const bool UsesSquaredDistance = false;
};

} // namespace mlpack

#endif
