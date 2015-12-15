/**
 * @file random.cpp
 *
 * Declarations of global random number generators.
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
