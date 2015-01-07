/**
 * @file epanechnikov_kernel.cpp
 * @author Neil Slagle
 *
 * Implementation of non-template Epanechnikov kernels.
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
double EpanechnikovKernel::Evaluate(const double distance) const
{
  return std::max(0.0, 1 - std::pow(distance, 2.0) * inverseBandwidthSquared);
}

// Return string of object.
std::string EpanechnikovKernel::ToString() const
{
  std::ostringstream convert;
  convert << "EpanechnikovKernel [" << this << "]" << std::endl;
  convert << "  Bandwidth: " << bandwidth << std::endl;
  convert << "  Inverse squared bandwidth: ";
  convert << inverseBandwidthSquared << std::endl;
  return convert.str();
}
