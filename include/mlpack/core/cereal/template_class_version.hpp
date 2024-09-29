/**
 * @file core/cereal/template_class_version.hpp
 * @author Ryan Curtin
 *
 * Implementation of CEREAL_TEMPLATE_CLASS_VERSION() macro, useful for
 * templatized types where CEREAL_CLASS_VERSION() will not work.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_CEREAL_TEMPLATE_CLASS_VERSION_HPP
#define MLPACK_CORE_CEREAL_TEMPLATE_CLASS_VERSION_HPP

#include <cereal/cereal.hpp>

// This useful implementation is adapted from @lubensky on Github:
// https://github.com/uscilab/cereal/issues/319#issuecomment-1512927210

#define CEREAL_UNPACK(...) __VA_ARGS__

#ifdef MLPACK_HAVE_CXX17

// The C++17 version sets `version` as `inline`.
#define CEREAL_TEMPLATE_CLASS_VERSION(ARGS, TYPE, VERSION_NUMBER)              \
namespace cereal {                                                             \
namespace detail {                                                             \
template<CEREAL_UNPACK ARGS>                                                   \
struct Version<CEREAL_UNPACK TYPE>                                             \
{                                                                              \
  static std::uint32_t registerVersion()                                       \
  {                                                                            \
    ::cereal::detail::StaticObject<Versions>::getInstance().mapping.emplace(   \
        std::type_index(typeid(CEREAL_UNPACK TYPE)).hash_code(),               \
                        CEREAL_UNPACK VERSION_NUMBER);                         \
    return CEREAL_UNPACK VERSION_NUMBER;                                       \
  }                                                                            \
                                                                               \
  static inline const std::uint32_t version = registerVersion();               \
                                                                               \
  static void unused() { (void) version; }                                     \
}; /* end Version */                                                           \
                                                                               \
}                                                                              \
}

#else

// Here we cannot use inline variables.
#define CEREAL_TEMPLATE_CLASS_VERSION(ARGS, TYPE, VERSION_NUMBER)              \
namespace cereal {                                                             \
namespace detail {                                                             \
template<CEREAL_UNPACK ARGS>                                                   \
struct Version<CEREAL_UNPACK TYPE>                                             \
{                                                                              \
  static const std::uint32_t version;                                          \
  static std::uint32_t registerVersion()                                       \
  {                                                                            \
    ::cereal::detail::StaticObject<Versions>::getInstance().mapping.emplace(   \
        std::type_index(typeid(CEREAL_UNPACK TYPE)).hash_code(),               \
                        CEREAL_UNPACK VERSION_NUMBER);                         \
    return CEREAL_UNPACK VERSION_NUMBER;                                       \
  }                                                                            \
                                                                               \
  static void unused() { (void) version; }                                     \
}; /* end Version */                                                           \
                                                                               \
template<CEREAL_UNPACK ARGS>                                                   \
const std::uint32_t Version<CEREAL_UNPACK TYPE>::version =                     \
    Version<CEREAL_UNPACK TYPE>::registerVersion();                            \
                                                                               \
}                                                                              \
}

#endif // MLPACK_HAVE_CXX17

#endif // TEMPLATE_CLASS_VERSION_HPP
