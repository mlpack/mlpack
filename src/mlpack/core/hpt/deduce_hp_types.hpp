/**
 * @file core/hpt/deduce_hp_types.hpp
 * @author Kirill Mishchenko
 *
 * Tools to deduce types of hyper-parameters from types of arguments in the
 * Optimize method in HyperParameterTuner.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_HPT_DEDUCE_HP_TYPES_HPP
#define MLPACK_CORE_HPT_DEDUCE_HP_TYPES_HPP

#include <mlpack/core/hpt/fixed.hpp>

namespace mlpack {

/**
 * A type function for deducing types of hyper-parameters from types of
 * arguments in the Optimize method in HyperParameterTuner.
 *
 * We start by putting all types of the arguments into Args, and then process
 * each of them one by one and put results into the internal struct
 * ResultHolder. By the end Args become empty, while ResultHolder holds the
 * tuple type of hyper-parameters.
 *
 * Here we declare and define DeduceHyperParameterTypes for the end phase when
 * Args are empty (all argument types have been processed).
 */
template<typename... Args>
struct DeduceHyperParameterTypes
{
  template<typename... HPTypes>
  struct ResultHolder
  {
    using TupleType = std::tuple<HPTypes...>;
  };
};

/**
 * Defining DeduceHyperParameterTypes for the case when not all argument types
 * have been processed, and the next one (T) is a collection type or an
 * arithmetic type.
 */
template<typename T, typename... Args>
struct DeduceHyperParameterTypes<T, Args...>
{
  /**
   * A type function to deduce the result hyper-parameter type for ArgumentType.
   */
  template<typename ArgumentType,
           bool IsArithmetic = std::is_arithmetic_v<ArgumentType>>
  struct ResultHPType;

  template<typename ArithmeticType>
  struct ResultHPType<ArithmeticType, true>
  {
    using Type = ArithmeticType;
  };

  /**
   * A type function to check whether Type is a collection type (for that it
   * should define value_type).
   */
  template<typename Type>
  struct IsCollectionType
  {
    using Yes = char[1];
    using No = char[2];

    template<typename TypeToCheck>
    static Yes& Check(typename TypeToCheck::value_type*);
    template<typename>
    static No& Check(...);

    static const bool value  =
      sizeof(decltype(Check<Type>(0))) == sizeof(Yes);
  };

  template<typename CollectionType>
  struct ResultHPType<CollectionType, false>
  {
    static_assert(IsCollectionType<CollectionType>::value,
        "One of the passed arguments is neither of an arithmetic type, nor of "
        "a collection type, nor fixed with the Fixed function.");

    using Type = typename CollectionType::value_type;
  };

  template<typename... HPTypes>
  struct ResultHolder
  {
    using TupleType = typename DeduceHyperParameterTypes<Args...>::template
        ResultHolder<HPTypes..., typename ResultHPType<T>::Type>::TupleType;
  };

  using TupleType = typename ResultHolder<>::TupleType;
};

/**
 * Defining DeduceHyperParameterTypes for the case when not all argument types
 * have been processed, and the next one is the type of an argument that should
 * be fixed.
 */
template<typename T, typename... Args>
struct DeduceHyperParameterTypes<PreFixedArg<T>, Args...>
{
  template<typename... HPTypes>
  struct ResultHolder
  {
    using TupleType = typename DeduceHyperParameterTypes<Args...>::template
        ResultHolder<HPTypes...>::TupleType;
  };

  using TupleType = typename ResultHolder<>::TupleType;
};

/**
 * A short alias for deducing types of hyper-parameters from types of arguments
 * in the Optimize method in HyperParameterTuner.
 */
template<typename... Args>
using TupleOfHyperParameters =
    typename DeduceHyperParameterTypes<Args...>::TupleType;

} // namespace mlpack

#endif
