/**
 * @file core/hpt/fixed.hpp
 * @author Kirill Mishchenko
 *
 * Facilities for supporting fixed arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_FIXED_HPP
#define MLPACK_CORE_HPT_FIXED_HPP

#include <type_traits>

#include <mlpack/core.hpp>

namespace mlpack {

template<typename>
struct PreFixedArg;

/**
 * Mark the given argument as one that should be fixed. It can be applied to
 * arguments that are passed to the Optimize method of HyperParameterTuner.
 *
 * The implementation avoids data copying. If the passed argument is an l-value
 * reference, we store it as a const l-value rerefence inside the returned
 * PreFixedArg object. If the passed argument is an r-value reference,
 * light-weight copying (by taking possession of the r-value) will be made
 * during the initialization of the returned PreFixedArg object.
 */
template<typename T>
PreFixedArg<T> Fixed(T&& value)
{
  return PreFixedArg<T>{std::forward<T>(value)};
}

/**
 * A struct for storing information about a fixed argument. Objects of this type
 * are supposed to be passed into the CVFunction constructor.
 *
 * This struct is not meant to be used directly by users. Rather use the `Fixed`
 * function.
 *
 * @tparam T The type of the fixed argument.
 * @tparam I The index of the fixed argument.
 */
template<typename T, size_t I>
struct FixedArg
{
  //! The index of the fixed argument.
  static const size_t index = I;

  //! The value of the fixed argument.
  const T& value;
};

/**
 * A struct for marking arguments as ones that should be fixed (it can be useful
 * for the Optimize method of HyperParameterTuner). Arguments of this type are
 * supposed to be converted into structs of the type FixedArg by adding
 * information about argument positions.
 *
 * This struct is not meant to be used directly by users. Rather use the `Fixed`
 * function.
 */
template<typename T>
struct PreFixedArg
{
  using Type = T;

  const T value;
};

/**
 * The specialization of the template for references.
 *
 * This struct is not meant to be used directly by users. Rather use the `Fixed`
 * function.
 */
template<typename T>
struct PreFixedArg<T&>
{
  using Type = T;

  const T& value;
};

/**
 * A type function for checking whether the given type is PreFixedArg.
 */
template<typename T>
class IsPreFixedArg
{
  template<typename>
  struct Implementation : std::false_type {};

  template<typename Type>
  struct Implementation<PreFixedArg<Type>> : std::true_type {};

 public:
  static const bool value = Implementation<std::decay_t<T>>::value;
};

} // namespace mlpack

#endif
