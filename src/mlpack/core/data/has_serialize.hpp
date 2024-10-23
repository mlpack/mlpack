/**
 * @file core/data/has_serialize.hpp
 * @author Ryan Curtin
 *
 * This file contains the HasSerialize struct, which can be used to detect
 * whether or not a type is serializable.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_HAS_SERIALIZE_HPP
#define MLPACK_CORE_UTIL_HAS_SERIALIZE_HPP

#include <mlpack/core/util/sfinae_utility.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>

#include <type_traits>

namespace mlpack {
namespace data {

// This gives us a HasSerializeCheck<T, U> type (where U is a function pointer)
// we can use with SFINAE to catch when a type has a Serialize() function.
HAS_EXACT_METHOD_FORM(serialize, HasSerializeCheck);

// Don't call this with a non-class.  HasSerializeFunction::value is true if the
// type T has a static or non-static Serialize() function.
template<typename T>
struct HasSerializeFunction
{
  template<typename C>
  using NonStaticSerialize = void(C::*)(cereal::XMLOutputArchive&,
      const uint32_t version);

  template<typename /* C */>
  using StaticSerialize = void(*)(cereal::XMLOutputArchive&,
      const uint32_t version);

  static const bool value = HasSerializeCheck<T, NonStaticSerialize>::value ||
                            HasSerializeCheck<T, StaticSerialize>::value;
};

template<typename T>
struct HasSerialize
{
  // We have to handle the case where T isn't a class...
  typedef char yes[1];
  typedef char no [2];
  template<typename U, typename V, typename W> struct check;
  template<typename U> static yes& chk( // This matches classes.
      check<U,
            typename std::enable_if_t<std::is_class_v<U>>*,
            typename std::enable_if_t<HasSerializeFunction<U>::value>*>*);
  template<typename  > static no&  chk(...); // This matches non-classes.

  static const bool value = (sizeof(chk<T>(0)) == sizeof(yes));
};

} // namespace data
} // namespace mlpack

#endif
