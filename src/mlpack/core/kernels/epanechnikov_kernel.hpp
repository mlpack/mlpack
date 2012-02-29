/**
 * @file epanechnikov_kernel.hpp
 * @author Neil Slagle
 *
 * This is an example kernel.  If you are making your own kernel, follow the
 * outline specified in this file.
 */
#ifndef __MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_H
#define __MLPACK_CORE_KERNELS_EPANECHNIKOV_KERNEL_H

#include <boost/math/special_functions/gamma.hpp>

#include <mlpack/core.hpp>
#include <mlpack/core/metrics/lmetric.hpp>

namespace mlpack {
namespace kernel {

class EpanechnikovKernel
{
 public:
  EpanechnikovKernel() :
    bandwidth(1.0),
    inverseBandwidthSquared(1.0) {}
  EpanechnikovKernel(double b) :
    bandwidth(b),
    inverseBandwidthSquared(1.0/(b*b)) {}

  template<typename VecType>
  double Evaluate(const VecType& a, const VecType& b)
  {
    double evaluatee =
      1.0 - metric::SquaredEuclideanDistance::Evaluate(a, b) * inverseBandwidthSquared;
    return (evaluatee > 0.0) ? evaluatee : 0.0;
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
        return 1.0 / volumeSquared *
          (16.0/15.0*bandwidth-4.0*distance*distance /
          (3.0*bandwidth)+2.0*distance*distance*distance/
          (3.0*bandwidth*bandwidth) -
          pow(distance,5.0)/(30.0*pow(bandwidth,4.0)));
        break;
      case 2:
        return 1.0 / volumeSquared *
          ((2.0/3.0*bandwidth*bandwidth-distance*distance)*
          asin(sqrt(1.0-pow(distance/(2.0*bandwidth),2.0))) +
          sqrt(4.0*bandwidth*bandwidth-distance*distance)*
          (distance/6.0+2.0/9.0*distance*pow(distance/bandwidth,2.0)-
          distance/72.0*pow(distance/bandwidth,4.0)));
        break;
      default:
        Log::Fatal << "Epanechnikov doesn't support your dimension (yet).";
        return -1.0;
        break;
    }
  }

  double Normalizer(size_t dimension)
  {
    return 2.0 * pow(bandwidth, dimension) * pow(M_PI, dimension / 2.0) /
             (tgamma(dimension / 2.0 + 1.0) * (dimension + 2.0));
  }
  double Evaluate(double t)
  {
    double evaluatee = 1.0 - t * t * inverseBandwidthSquared;
    return (evaluatee > 0.0) ? evaluatee : 0.0;
  }
 private:
  double bandwidth;
  double inverseBandwidthSquared;
};

}; // namespace kernel
}; // namespace mlpack

#endif
