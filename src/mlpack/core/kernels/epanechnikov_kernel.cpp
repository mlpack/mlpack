/**
 * @file epanechnikov_kernel.hpp
 * @author Neil Slagle
 *
 * This is an example kernel.  If you are making your own kernel, follow the
 * outline specified in this file.
 */
#include "epanechnikov_kernel.hpp"

#include <boost/math/special_functions/gamma.hpp>

using namespace mlpack;
using namespace mlpack::kernel;

/**
 * Compute the normalizer of this Epanechnikov kernel for the given dimension.
 *
 * @param dimension Dimension to calculate the normalizer for.
 */
double EpanechnikovKernel::Normalizer(const size_t dimension)
{
  return 2.0 * pow(bandwidth, (double) dimension) *
      std::pow(M_PI, dimension / 2.0) /
      (boost::math::tgamma(dimension / 2.0 + 1.0) * (dimension + 2.0));
}

/**
 * Evaluate the kernel not for two points but for a numerical value.
 */
double EpanechnikovKernel::Evaluate(const double t)
{
  double evaluatee = 1.0 - t * t * inverseBandwidthSquared;
  return (evaluatee > 0.0) ? evaluatee : 0.0;
}
