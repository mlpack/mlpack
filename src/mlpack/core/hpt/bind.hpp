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

template<typename>
struct PreBoundArg;

/**
 * Mark the given argument as one that should be bound. It can be applied to
 * arguments that are passed to the Optimize method of HyperParameterTuner.
 *
 * The implementation avoids data copying. If the passed argument is an l-value
 * reference, we store it as a const l-value rerefence inside the returned
 * PreBoundArg object. If the passed argument is an r-value reference,
 * ligth-weight coping (by taking possesion of the r-value) will be made during
 * the initialization of the returned PreBoundArg object.
 */
template<typename T>
PreBoundArg<T> Bind(T&& value)
{
  return PreBoundArg<T>{std::forward<T>(value)};
}

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

/**
 * A struct for marking arguments as ones that should be bound (it can be useful
 * for the Optimize method of HyperParameterTuner). Arguments of this type are
 * supposed to be converted into structs of the type BoundArg by adding
 * information about argument positions.
 *
 * This struct is not meant to be used directly by users. Rather use the
 * mlpack::hpt::Bind function.
 */
template<typename T>
struct PreBoundArg
{
  using Type = T;

  const T value;
};

/**
 * The specialization of the template for references.
 *
 * This struct is not meant to be used directly by users. Rather use the
 * mlpack::hpt::Bind function.
 */
template<typename T>
struct PreBoundArg<T&>
{
  using Type = T;

  const T& value;
};

/**
 * A type function for checking whether the given type is PreBoundArg.
 */
template<typename T>
class IsPreBoundArg
{
  template<typename>
  struct Implementation : std::false_type {};

  template<typename Type>
  struct Implementation<PreBoundArg<Type>> : std::true_type {};

 public:
  static const bool value = Implementation<typename std::decay<T>::type>::value;
};

} // namespace hpt
} // namespace mlpack

#endif
