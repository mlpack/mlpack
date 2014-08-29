/**
 * @file hyperbolic_tangent_kernel.hpp
 * @author Ajinkya Kale <kaleajinkya@gmail.com>
 *
 * Implementation of the hyperbolic tangent kernel.
 *
 * This file is part of MLPACK 1.0.10.
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
#ifndef __MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP
#define __MLPACK_CORE_KERNELS_HYPERBOLIC_TANGENT_KERNEL_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

/**
 * Hyperbolic tangent kernel.  For any two vectors @f$ x @f$, @f$ y @f$ and a
 * given scale @f$ s @f$ and offset @f$ t @f$
 *
 * @f[
 * K(x, y) = \tanh(s <x, y> + t)
 * @f]
 */
class HyperbolicTangentKernel
{
 public:
  /**
   * This constructor sets the default scale to 1.0 and offset to 0.0.
   */
  HyperbolicTangentKernel() : scale(1.0), offset(0.0)
  { }

  /**
   * Construct the hyperbolic tangent kernel with custom scale factor and
   * offset.
   *
   * @param scale Scaling factor for <x, y>.
   * @param offset Kernel offset.
   */
  HyperbolicTangentKernel(double scale, double offset) :
      scale(scale), offset(offset)
  { }

  /**
   * Evaluate the hyperbolic tangent kernel.  This evaluation uses Armadillo's
   * dot() function.
   *
   * @tparam VecType Type of vector (should be arma::vec or arma::spvec).
   * @param a First vector.
   * @param b Second vector.
   * @return K(a, b).
   */
  template<typename VecType>
  double Evaluate(const VecType& a, const VecType& b)
  {
    return tanh(scale * arma::dot(a, b) + offset);
  }

  //! Get scale factor.
  double Scale() const { return scale; }
  //! Modify scale factor.
  double& Scale() { return scale; }

  //! Get offset for the kernel.
  double Offset() const { return offset; }
  //! Modify offset for the kernel.
  double& Offset() { return offset; }

  //! Convert object to string.
  std::string ToString() const
  {
    std::ostringstream convert;
    convert << "HyperbolicTangentKernel [" << this << "]" << std::endl;
    convert << "  Scale: " << scale << std::endl;
    convert << "  Offset: " << offset << std::endl;
    return convert.str();
  }

 private:
  double scale;
  double offset;
};

}; // namespace kernel
}; // namespace mlpack

#endif
