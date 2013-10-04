/**
 * @file spherical_kernel.hpp
 * @author Neil Slagle
 *
 * This is an example kernel.  If you are making your own kernel, follow the
 * outline specified in this file.
 *
 * This file is part of MLPACK 1.0.7.
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
#ifndef __MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_H
#define __MLPACK_CORE_KERNELS_SPHERICAL_KERNEL_H

#include <boost/math/special_functions/gamma.hpp>
#include <mlpack/core.hpp>

namespace mlpack {
namespace kernel {

class SphericalKernel
{
 public:
  SphericalKernel() :
    bandwidth(1.0),
    bandwidthSquared(1.0) {}
  SphericalKernel(double b) :
    bandwidth(b),
    bandwidthSquared(b*b) {}

  template<typename VecType>
  double Evaluate(const VecType& a, const VecType& b)
  {
    return
      (metric::SquaredEuclideanDistance::Evaluate(a, b) <= bandwidthSquared) ?
        1.0 : 0.0;
  }
  /**
   * Obtains the convolution integral [integral K(||x-a||)K(||b-x||)dx]
   * for the two vectors.  In this case, because
   * our simple example kernel has no internal parameters, we can declare the
   * function static.  For a more complex example which cannot be declared
   * static, see the GaussianKernel, which stores an internal parameter.
   *
   * @tparam VecType Type of vector (arma::vec, arma::spvec should be expected).
   * @param a First vector.
   * @param b Second vector.
   * @return the convolution integral value.
   */
  template<typename VecType>
  double ConvolutionIntegral(const VecType& a, const VecType& b)
  {
    double distance = sqrt(metric::SquaredEuclideanDistance::Evaluate(a, b));
    if (distance >= 2.0 * bandwidth)
    {
      return 0.0;
    }
    double volumeSquared = pow(Normalizer(a.n_rows), 2.0);

    switch(a.n_rows)
    {
      case 1:
        return 1.0 / volumeSquared * (2.0 * bandwidth - distance);
        break;
      case 2:
        return 1.0 / volumeSquared *
          (2.0 * bandwidth * bandwidth * acos(distance/(2.0 * bandwidth)) -
          distance / 4.0 * sqrt(4.0*bandwidth*bandwidth-distance*distance));
        break;
      default:
        Log::Fatal << "The spherical kernel does not support convolution\
          integrals above dimension two, yet..." << std::endl;
        return -1.0;
        break;
    }
  }
  double Normalizer(size_t dimension)
  {
    return pow(bandwidth, (double) dimension) * pow(M_PI, dimension / 2.0) /
        boost::math::tgamma(dimension / 2.0 + 1.0);
  }
  double Evaluate(double t)
  {
    return (t <= bandwidth) ? 1.0 : 0.0;
  }

 private:
  double bandwidth;
  double bandwidthSquared;
};

//! Kernel traits for the spherical kernel.
template<>
class KernelTraits<SphericalKernel>
{
 public:
  //! The spherical kernel is normalized: K(x, x) = 1 for all x.
  static const bool IsNormalized = true;
};

}; // namespace kernel
}; // namespace mlpack

#endif
