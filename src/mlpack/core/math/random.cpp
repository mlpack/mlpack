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
  boost::random::uniform_01<> randUniformDist;
  // Global normal distribution.
  boost::random::normal_distribution<> randNormalDist;
#else
  // Global random object.
  boost::mt19937 randGen;

  #if BOOST_VERSION >= 103900
    // Global uniform distribution.
    boost::uniform_01<> randUniformDist;
  #else
    // Pre-1.39 Boost.Random did not give default template parameter values.
    boost::uniform_01<double, double> randUniformDist;
  #endif

  // Global normal distribution.
  boost::normal_distribution<> randNormalDist;
#endif

}; // namespace math
}; // namespace mlpack
