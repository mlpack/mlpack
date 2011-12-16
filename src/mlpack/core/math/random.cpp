/**
 * @file random.cpp
 *
 * Declarations of global Boost random number generators.
 */
#include <boost/random.hpp>
#include <boost/version.hpp>

namespace mlpack {
namespace math {

#if BOOST_VERSION >= 104700
  // Global random object.
  boost::random::mt19937 randGen;
  // Global uniform distribution.
  boost::random::uniform_01 randUniformDist;
#else
  // Global random object.
  boost::mt19937 randGen;
  // Global uniform distribution.
  boost::uniform_01<> randUniformDist;
#endif

}; // namespace math
}; // namespace mlpack
