/**
 * @file boost_backport.hpp
 * @author Yannis Mentekidis
 *
 * Centralized control of what boost files to include. We have backported the
 * following boost functionality:
 *
 *  * unordered_set serialization support (added in boost 1.56.0)
 *  * trigamma and polygamma function evaluation (added in boost 1.58.0)
 *
 * If the detected boost version is greater than 1.58.0, we include the normal
 * serialization, polygamma and trigamma functions (not the backported ones). 
 * For versions 1.56, 1.57 we include the normal serialization but the 
 * backported polygamma and trigamma functions.
 * For all older versions we include the backported headers.
 */

#ifndef MLPACK_CORE_BOOST_BACKPORT_HPP
#define MLPACK_CORE_BOOST_BACKPORT_HPP

#include <boost/version.hpp>

#if BOOST_VERSION < 105600
  // Backported unordered_map.
  #include "mlpack/core/boost_backport/unordered_map.hpp"
#else
  // Boost's version
  #include <boost/serialization/unordered_map.hpp>
#endif

#if BOOST_VERSION < 105800
  // Backported trigamma and polygamma.
  #include "mlpack/core/boost_backport/trigamma.hpp"
  #include "mlpack/core/boost_backport/polygamma.hpp"
#else
  // Boost's version.
  #include <boost/math/special_functions/trigamma.hpp>
  #include <boost/math/special_functions/polygamma.hpp>
#endif

#endif // MLPACK_CORE_BOOST_BACKPORT_HPP

