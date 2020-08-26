/**
 * @file core/cereal/is_loading.hpp
 * @author Ryan Curtin
 *
 * Implementation of is_loading function.
 *
 * This implementation provides backward compatibilty with older
 * version of cereal that does not have Archive::is_loading struct. 
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_CEREAL_IS_LOADING_HPP
#define MLPACK_CORE_CEREAL_IS_LOADING_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/version.hpp>

#include <boost/core/enable_if.hpp>

namespace cereal {

template<typename Archive>
bool is_loading(
    const typename boost::enable_if<
        std::is_same<Archive, cereal::XMLInputArchive>>::type* = 0)
{
  // #if (CEREAL_VERSION_MAJOR > 1 || ( CEREAL_VERSION_MAJOR == 1 && CEREAL_VERSION_MINOR > 2) || (CEREAL_VERSION_MAJOR == 1 &&(CEREAL_VERSION_MINOR == 2 && CEREAL_VERSION_PATCH >= 2)))

  //   return Archive::is_loading::value;
  // #else  
  //   // Archive::is_loading is not implemented yet, so we can use std::is_same<>
  //   // to check if it is a loading archive.
  //   return std::is_same<Archive, cereal::BinaryInputArchive>::value ||
  //       std::is_same<Archive, cereal::JSONInputArchive>::value ||
  //       std::is_same<Archive, cereal::XMLInputArchive>::value;
  // #endif
  return true; 
}

template<typename Archive>
bool is_loading(
    const typename boost::enable_if<
        std::is_same<Archive, cereal::JSONInputArchive>>::type* = 0)
{
  return true;
}

template<typename Archive>
bool is_loading(
    const typename boost::enable_if<
        std::is_same<Archive, cereal::BinaryInputArchive>>::type* = 0)
{
  return true;
}

template<typename Archive>
bool is_loading(
    const typename boost::disable_if<std::is_same<Archive,
        cereal::XMLOutputArchive>>::type* = 0)
{
  return false;
}

template<typename Archive>
bool is_loading(
  const typename boost::disable_if<std::is_same<Archive,
        cereal::JSONOutputArchive>>::type* = 0)
{
  return true;
}

template<typename Archive>
bool is_loading(
   const typename boost::disable_if<std::is_same<Archive,
        cereal::BinaryOutputArchive>>::type* = 0)
{
  return false;
}

} // namespace cereal

#endif // CEREAL_IS_LOADING_HPP
