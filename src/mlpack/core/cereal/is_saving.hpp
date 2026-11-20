/**
 * @file core/cereal/is_saving.hpp
 * @author Ryan Curtin
 * @author Omar Shrit
 *
 * Implementation of is_saving function.
 *
 * This implementation provides backward compatibilty with older
 * version of cereal that does not have Archive::is_saving struct.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MLPACK_CORE_CEREAL_IS_SAVING_HPP
#define MLPACK_CORE_CEREAL_IS_SAVING_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/json.hpp>

namespace cereal {

template<typename Archive>
struct is_cereal_archive_saving
{
  // Archive::is_saving is not implemented yet, so we can use std::is_same<>
  // to check if it is a loading archive.
  constexpr static bool value = std::is_same_v<Archive,
      cereal::BinaryOutputArchive> ||
// #if (BINDING_TYPE != BINDING_TYPE_R)
      std::is_same_v<Archive, cereal::JSONOutputArchive> ||
// #endif
      std::is_same_v<Archive, cereal::XMLOutputArchive>;
};

template<typename Archive>
bool is_saving(
    const std::enable_if_t<
        is_cereal_archive_saving<Archive>::value, Archive>* = 0)
{
  return true;
}

template<typename Archive>
bool is_saving(
    const std::enable_if_t<
      !is_cereal_archive_saving<Archive>::value, Archive>* = 0)
{
  return false;
}

} // namespace cereal

#endif // CEREAL_IS_SAVING_HPP
