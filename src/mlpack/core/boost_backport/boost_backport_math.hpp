/**
 * @file boost_backport.hpp
 * @author Yannis Mentekidis
 *
 * Centralized control of what boost files to include. We have backported the
 * following boost functionality here:
 *
 *  * trigamma and polygamma function evaluation (added in boost 1.58.0)
 *
 * For versions 1.56, 1.57 we include the backported polygamma and trigamma
 * functions.  Anything newer, we include from Boost.
 */
#ifndef MLPACK_CORE_BOOST_BACKPORT_MATH_HPP
#define MLPACK_CORE_BOOST_BACKPORT_MATH_HPP

#include <boost/version.hpp>

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

