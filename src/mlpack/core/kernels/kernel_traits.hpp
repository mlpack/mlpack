/**
 * @file kernel_traits.hpp
 * @author Ryan Curtin
 *
 * This provides the KernelTraits class, a template class to get information
 * about various kernels.
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
#ifndef __MLPACK_CORE_KERNELS_KERNEL_TRAITS_HPP
#define __MLPACK_CORE_KERNELS_KERNEL_TRAITS_HPP

namespace mlpack {
namespace kernel {

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
};

}; // namespace kernel
}; // namespace mlpack

#endif
