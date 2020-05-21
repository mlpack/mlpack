/**
 * @file core/boost_backport/boost_backport_string_view.hpp
 * @author Jeffin Sam
 *
 * Centralized control of what boost files to include. We have backported the
 * following boost functionality here:
 *
 *  * string_view support (added in boost 1.61.0)
 *  * hash function support (added in boost 1.69.0)
 *
 * If the detected boost version is greater or equal to 1.61.0, we include the
 * normal serialization functions (not the backported ones).  For all older
 *  versions we include the backported headers.
 */
#ifndef MLPACK_CORE_BOOST_BACKPORT_STRING_VIEW_HPP
#define MLPACK_CORE_BOOST_BACKPORT_STRING_VIEW_HPP

#include <boost/version.hpp>
#include <boost/functional/hash.hpp>

#if BOOST_VERSION < 106100
  // Backported unordered_map.
  #include "mlpack/core/boost_backport/string_view.hpp"
#else
  // Boost's version.
  #include <boost/utility/string_view.hpp>
#endif

#if BOOST_VERSION < 106900
  namespace boost
  {
    template<>
    struct hash<boost::string_view>
    {
      std::size_t operator()(boost::string_view str) const
      {
        return boost::hash_range(str.begin(), str.end());
      }
    };
  }
#endif

#endif // MLPACK_CORE_BOOST_BACKPORT_STRING_VIEW_HPP
