/**
 * @file random.cpp
 *
 * Declarations of global random number generators.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <random>
#include <mlpack/mlpack_export.hpp>

namespace mlpack {
namespace math {

// Global random object.
MLPACK_EXPORT std::mt19937 randGen;
// Global uniform distribution.
MLPACK_EXPORT std::uniform_real_distribution<> randUniformDist(0.0, 1.0);
// Global normal distribution.
MLPACK_EXPORT std::normal_distribution<> randNormalDist(0.0, 1.0);

} // namespace math
} // namespace mlpack
