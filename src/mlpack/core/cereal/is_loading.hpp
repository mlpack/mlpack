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

namespace cereal {

template<typename Archive>
struct is_cereal_archive
{
  // Archive::is_loading is not implemented yet, so we can use std::is_same<>
  // to check if it is a loading archive.
  constexpr static bool value = std::is_same_v<Archive,
      cereal::BinaryInputArchive> ||
// #if (BINDING_TYPE != BINDING_TYPE_R)
      std::is_same_v<Archive, cereal::JSONInputArchive> ||
// #endif
      std::is_same_v<Archive, cereal::XMLInputArchive>;
};

template<typename Archive>
bool is_loading(
    const std::enable_if_t<is_cereal_archive<Archive>::value, Archive>* = 0)
{
  return true;
}

template<typename Archive>
bool is_loading(
    const std::enable_if_t<!is_cereal_archive<Archive>::value, Archive>* = 0)
{
  return false;
}

} // namespace cereal

#endif // CEREAL_IS_LOADING_HPP
