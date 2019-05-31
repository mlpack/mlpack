/**
 * @file boost_backport_string_view.hpp
 * @author Jeffin sam
 *
 * Centralized control of what boost files to include. We have backported the
 * following boost functionality here:
 *
 *  * string_view support (added in boost 1.61.0)
 *
 * If the detected boost version is greater than 1.61.0, we include the normal
 * serialization functions (not the backported ones).  For all older versions we
 * include the backported headers.
 */
#ifndef MLPACK_CORE_BOOST_BACKPORT_STRING_VIEW_HPP
#define MLPACK_CORE_BOOST_BACKPORT_STRING_VIEW_HPP

#include <boost/version.hpp>

#if BOOST_VERSION < 106100
  // Backported unordered_map.
  #include "mlpack/core/boost_backport/string_view.hpp"
#else
  // Boost's version.
  #include <boost/utility/string_view.hpp>
#endif

namespace boost
{
  template<>
  struct hash<boost::string_view>
  {
    std::size_t operator()(boost::string_view str) const
    {
      return boost::hash_range<const char*>(str.begin(), str.end());
    }
  };
}

#endif // MLPACK_CORE_BOOST_BACKPORT_STRING_VIEW_HPP