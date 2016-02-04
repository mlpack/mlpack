/**
 * @file random.cpp
 *
 * Declarations of global random number generators.
 *
 * This file is part of mlpack 2.0.1.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <random>

namespace mlpack {
namespace math {

// Global random object.
std::mt19937 randGen;
// Global uniform distribution.
std::uniform_real_distribution<> randUniformDist(0.0, 1.0);
// Global normal distribution.
std::normal_distribution<> randNormalDist(0.0, 1.0);

} // namespace math
} // namespace mlpack
