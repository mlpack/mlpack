/**
 * @file bind.hpp
 * @author Kirill Mishchenko
 *
 * Facilities for supporting bound arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_BIND_HPP
#define MLPACK_CORE_HPT_BIND_HPP

#include <type_traits>

#include <mlpack/core.hpp>

namespace mlpack {
namespace hpt {

/**
 * A struct for storing information about a bound argument. Objects of this type
 * are supposed to be passed into the CVFunction constructor.
 *
 * This struct is not meant to be used directly by users. Rather use the
 * mlpack::hpt::Bind function.
 *
 * @tparam T The type of the bound argument.
 * @tparam I The index of the bound argument.
 */
template<typename T, size_t I>
struct BoundArg
{
  //! The index of the bound argument.
  static const size_t index = I;

  //! The value of the bound argument.
  const T& value;
};

} // namespace hpt
} // namespace mlpack

#endif
