/**
 * @file sfinae_utility.hpp
 * @author Trironk Kiatkungwanglai
 *
 * This file contains macro utilities for the SFINAE Paradigm. These utilities
 * determine if classes passed in as template parameters contain members at
 * compile time, which is useful for changing functionality depending on what
 * operations an object is capable of performing.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_SFINAE_UTILITY
#define MLPACK_CORE_SFINAE_UTILITY

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>

/*
 * Constructs a template supporting the SFINAE pattern.
 *
 * This macro generates a template struct that is useful for enabling/disabling
 * a method if the template class passed in contains a member function matching
 * a given signature with a specified name.
 *
 * The generated struct should be used in conjunction with boost::disable_if and
 * boost::enable_if. Here is an example usage:
 *
 * For general references, see:
 * http://stackoverflow.com/a/264088/391618
 *
 * For an mlpack specific use case, see /mlpack/core/util/prefixedoutstream.hpp
 * and /mlpack/core/util/prefixedoutstream_impl.hpp
 *
 * @param NAME the name of the struct to construct. For example: HasToString
 * @param FUNC the name of the function to check for. For example: ToString
 */
#define HAS_MEM_FUNC(FUNC, NAME)                                               \
template<typename T, typename sig>                                             \
struct NAME {                                                                  \
  typedef char yes[1];                                                         \
  typedef char no [2];                                                         \
  template<typename U, U> struct type_check;                                   \
  template<typename _1> static yes &chk(type_check<sig, &_1::FUNC> *);         \
  template<typename   > static no  &chk(...);                                  \
  static bool const value = sizeof(chk<T>(0)) == sizeof(yes);                  \
};

#endif
