/**
 * @file boost_backport.hpp
 * @author Yannis Mentekidis
 *
 * Centralized control of what boost files to include. We have backported the
 * following boost functionality here:
 *
 *  * unordered_set serialization support (added in boost 1.56.0)
 *
 * If the detected boost version is greater than 1.58.0, we include the normal
 * serialization functions (not the backported ones).  For all older versions we
 * include the backported headers.
 */
#ifndef MLPACK_CORE_BOOST_BACKPORT_SERIALIZATION_HPP
#define MLPACK_CORE_BOOST_BACKPORT_SERIALIZATION_HPP

#include <boost/version.hpp>

#if BOOST_VERSION < 105600
  // Backported unordered_map.
  #include "mlpack/core/boost_backport/unordered_map.hpp"
#else
  // Boost's version.
  #include <boost/serialization/unordered_map.hpp>
#endif

#endif // MLPACK_CORE_BOOST_BACKPORT_SERIALIZATION_HPP

