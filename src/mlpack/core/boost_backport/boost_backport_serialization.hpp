/**
 * @file boost_backport.hpp
 * @author Yannis Mentekidis
 *
 * Centralized control of what boost files to include. We have backported the
 * following boost functionality here:
 *
 *  * unordered_set serialization support (added in boost 1.56.0)
 *  * vector serialization (changed after boost 1.58.0)
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

#if BOOST_VERSION == 105800
  /**
   * Boost versions 1.58.0 and earlier have a different vector serialization
   * behaivor as compared to later versions. Notably, loading a
   * std::vector<arma::mat> does not clear the vector before the load
   * in v1.58 and earlier; while in the later versions, the vector is cleared
   * before loading. This causes some tests related to serialization to fail
   * with versions 1.58. This backport solves the issue.
   */
  #ifdef BOOST_SERIALIZATION_VECTOR_HPP
  #pragma message "Detected Boost version is 1.58. Including\
  boost/serialization/vector.hpp before mlpack/core.hpp can cause problems. It\
  should only be necessary to include mlpack/core.hpp and not\
  boost/serialization/vector.hpp."
  #endif
  #include "mlpack/core/boost_backport/collections_load_imp.hpp"
  #include "mlpack/core/boost_backport/collections_save_imp.hpp"
  #include "mlpack/core/boost_backport/vector.hpp"
#else
  #include <boost/serialization/vector.hpp>
#endif

#endif // MLPACK_CORE_BOOST_BACKPORT_SERIALIZATION_HPP

