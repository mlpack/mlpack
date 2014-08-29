/**
 * @file random.cpp
 *
 * Declarations of global Boost random number generators.
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
    boost::uniform_01<boost::mt19937, double> randUniformDist(randGen);
  #endif

  // Global normal distribution.
  boost::normal_distribution<> randNormalDist;
#endif

}; // namespace math
}; // namespace mlpack
