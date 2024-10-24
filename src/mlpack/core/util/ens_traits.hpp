/**
 * @file core/util/ens_traits.hpp
 * @author Ryan Curtin
 *
 * This file contains utilities for SFINAE on ensmallen types (optimizers and
 * callbacks).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_ENS_TRAITS
#define MLPACK_CORE_UTIL_ENS_TRAITS

#include "sfinae_utility.hpp"

namespace mlpack {

HAS_MEM_FUNC(Optimize, HasOptimize);

// Utility struct for IsEnsOptimizer (below).  By default returns false.  (See
// specialization below for the real logic.)
template<typename OptimizerType,
         typename FunctionType,
         typename MatType,
         bool OptimizerTypeIsClass>
struct IsEnsOptimizerInternal
{
  constexpr static bool value = false;
};

/**
 * If the given argument is an ensmallen-compatible optimizer for the given
 * MatType and FunctionType (e.g. if it has an `Optimize()` function that can
 * handle the given `FunctionType` and `MatType`), then the `value` member will
 * be `true`.
 */
template<typename OptimizerType, typename FunctionType, typename MatType>
struct IsEnsOptimizer
{
  // Dispatch to IsEnsOptimizerInternal, which will filter out when
  // OptimizerType is a non-class.
  constexpr static bool value = IsEnsOptimizerInternal<
      OptimizerType,
      FunctionType,
      MatType,
      std::is_class_v<OptimizerType>
  >::value;
};

// Logic for detecting ensmallen optimizers when OptimizerType is a class.
template<typename OptimizerType, typename FunctionType, typename MatType>
struct IsEnsOptimizerInternal<OptimizerType, FunctionType, MatType, true>
{
  // If OptimizerType is a reference type, then forming the types below will
  // fail.  So we need to strip the reference (and the const for good measure).
  using SafeOptimizerType =
      std::remove_cv_t<std::remove_reference_t<OptimizerType>>;

  using OptimizeElemReturnForm =
      typename MatType::elem_type(SafeOptimizerType::*)(FunctionType&,
                                                        MatType&);

  using OptimizeVoidReturnForm =
      void(SafeOptimizerType::*)(FunctionType&, MatType&);

  constexpr static bool value =
      HasOptimize<SafeOptimizerType, OptimizeElemReturnForm>::value ||
      HasOptimize<SafeOptimizerType, OptimizeVoidReturnForm>::value;
};

/**
 * If the given template parameter pack could all be valid ensmallen callbacks
 * (i.e. they are all classes), then the `value` member will be `true`.
 *
 * This is not a perfect check, but it is sufficient to differentiate from
 * hyperparameters.
 */
template<typename... CallbackTypes>
struct IsEnsCallbackTypes;

template<typename CallbackType, typename... CallbackTypes>
struct IsEnsCallbackTypes<CallbackType, CallbackTypes...>
{
  constexpr static bool value =
      std::is_class_v<std::remove_cv_t<std::remove_reference_t<CallbackType>>>
      && IsEnsCallbackTypes<CallbackTypes...>::value;
};

template<>
struct IsEnsCallbackTypes<>
{
  constexpr static bool value = true;
};

} // namespace mlpack

#endif
