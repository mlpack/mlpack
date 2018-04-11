/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file provides metaprogramming utilities for detecting certain members of
 * FunctionType classes.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_TRAITS_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_TRAITS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {
namespace optimization {
namespace traits {

//! Detect an Evaluate() method.
HAS_EXACT_METHOD_FORM(Evaluate, HasEvaluate);
//! Detect a Gradient() method.
HAS_EXACT_METHOD_FORM(Gradient, HasGradient);
//! Detect an EvaluateWithGradient() method.
HAS_EXACT_METHOD_FORM(EvaluateWithGradient, HasEvaluateWithGradient);
//! Detect a NumFunctions() method.
HAS_EXACT_METHOD_FORM(NumFunctions, HasNumFunctions);
//! Detect a Shuffle() method.
HAS_EXACT_METHOD_FORM(Shuffle, HasShuffle);
//! Detect a NumConstraints() method.
HAS_EXACT_METHOD_FORM(NumConstraints, HasNumConstraints);
//! Detect an EvaluateConstraint() method.
HAS_EXACT_METHOD_FORM(EvaluateConstraint, HasEvaluateConstraint);
//! Detect a GradientConstraint() method.
HAS_EXACT_METHOD_FORM(GradientConstraint, HasGradientConstraint);
//! Detect a NumFeatures() method.
HAS_EXACT_METHOD_FORM(NumFeatures, HasNumFeatures);
//! Detect a PartialGradient() method.
HAS_EXACT_METHOD_FORM(PartialGradient, HasPartialGradient);

//! This is the form of a non-const Evaluate() method.
template<typename FunctionType>
using EvaluateForm = double(FunctionType::*)(const arma::mat&);

//! This is the form of a const Evaluate() method.
template<typename FunctionType>
using EvaluateConstForm =
    double(FunctionType::*)(const arma::mat&) const;

//! This is the form of a static Evaluate() method.
template<typename FunctionType>
using EvaluateStaticForm = double(*)(const arma::mat&);

//! This is the form of a non-const Gradient() method.
template<typename FunctionType>
using GradientForm = void(FunctionType::*)(const arma::mat&, arma::mat&);

//! This is the form of a const Gradient() method.
template<typename FunctionType>
using GradientConstForm =
    void(FunctionType::*)(const arma::mat&, arma::mat&) const;

//! This is the form of a static Gradient() method.
template<typename FunctionType>
using GradientStaticForm = void(*)(const arma::mat&, arma::mat&);

//! This is the form of a non-const EvaluateWithGradient() method.
template<typename FunctionType>
using EvaluateWithGradientForm =
    double(FunctionType::*)(const arma::mat&, arma::mat&);

//! This is the form of a const EvaluateWithGradient() method.
template<typename FunctionType>
using EvaluateWithGradientConstForm =
    double(FunctionType::*)(const arma::mat&, arma::mat&) const;

//! This is the form of a static EvaluateWithGradient() method.
template<typename FunctionType>
using EvaluateWithGradientStaticForm =
    double(*)(const arma::mat&, arma::mat&);

//! This is the form of a non-const NumFunctions() method.
template <typename FunctionType>
using NumFunctionsForm = size_t(FunctionType::*)();

//! This is the form of a const NumFunctions() method.
template <typename FunctionType>
using NumFunctionsConstForm = size_t(FunctionType::*)() const;

//! This is the form of a static NumFunctions() method.
template<typename FunctionType>
using NumFunctionsStaticForm = size_t(*)();

//! This is the form of a non-const Shuffle() method.
template<typename FunctionType>
using ShuffleForm = void(FunctionType::*)();

//! This is the form of a const Shuffle() method.
template<typename FunctionType>
using ShuffleConstForm = void(FunctionType::*)() const;

//! This is the form of a static Shuffle() method.
template<typename FunctionType>
using ShuffleStaticForm = void(*)();

//! This is the form of a decomposable Evaluate() method.
template<typename FunctionType>
using DecomposableEvaluateForm = double(FunctionType::*)(
    const arma::mat&, const size_t, const size_t);

//! This is the form of a decomposable const Evaluate() method.
template<typename FunctionType>
using DecomposableEvaluateConstForm = double(FunctionType::*)(
    const arma::mat&, const size_t, const size_t) const;

//! This is the form of a decomposable static Evaluate() method.
template<typename FunctionType>
using DecomposableEvaluateStaticForm = double(*)(
    const arma::mat&, const size_t, const size_t);

//! This is the form of a decomposable non-const Gradient() method.
template<typename FunctionType>
using DecomposableGradientForm = void(FunctionType::*)(
    const arma::mat&, const size_t, arma::mat&, const size_t);

//! This the form of a decomposable const Gradient() method.
template<typename FunctionType>
using DecomposableGradientConstForm = void(FunctionType::*)(
    const arma::mat&, const size_t, arma::mat&, const size_t) const;

//! This is the form of a decomposable static Gradient() method.
template<typename FunctionType>
using DecomposableGradientStaticForm = void(*)(
    const arma::mat&, const size_t, arma::mat&, const size_t);

//! This is the form of a decomposable non-const EvaluateWithGradient() method.
template<typename FunctionType>
using DecomposableEvaluateWithGradientForm = double(FunctionType::*)(
    const arma::mat&, const size_t, arma::mat&, const size_t);

//! This is the form of a decomposable const EvaluateWithGradient() method.
template<typename FunctionType>
using DecomposableEvaluateWithGradientConstForm = double(FunctionType::*)(
    const arma::mat&, const size_t, arma::mat&, const size_t) const;

//! This is the form of a decomposable static EvaluateWithGradient() method.
template<typename FunctionType>
using DecomposableEvaluateWithGradientStaticForm = double(*)(
    const arma::mat&, const size_t, arma::mat&, const size_t);

//! This is the form of a non-const NumConstraints() method.
template<typename FunctionType>
using NumConstraintsForm = size_t(FunctionType::*)();

//! This is the form of a const NumConstraints() method.
template<typename FunctionType>
using NumConstraintsConstForm = size_t(FunctionType::*)() const;

//! This is the form of a static NumConstraints() method.
template<typename FunctionType>
using NumConstraintsStaticForm = size_t(*)();

//! This is the form of a non-const EvaluateConstraint() method.
template <typename FunctionType>
using EvaluateConstraintForm = double(FunctionType::*)(
    const size_t, const arma::mat&);

//! This is the form of a const EvaluateConstraint() method.
template<typename FunctionType>
using EvaluateConstraintConstForm = double(FunctionType::*)(
    const size_t, const arma::mat&) const;

//! This is the form of a static EvaluateConstraint() method.
template<typename FunctionType>
using EvaluateConstraintStaticForm = double(*)(const size_t, const arma::mat&);

//! This is the form of a non-const GradientConstraint() method.
template <typename FunctionType>
using GradientConstraintForm = void(FunctionType::*)(
    const size_t, const arma::mat&, arma::mat&);

//! This is the form of a const GradientConstraint() method.
template<typename FunctionType>
using GradientConstraintConstForm = void(FunctionType::*)(
    const size_t, const arma::mat&, arma::mat&) const;

//! This is the form of a static GradientConstraint() method.
template<typename Class, typename... Ts>
using GradientConstraintStaticForm = void(*)(
    const size_t, const arma::mat&, arma::mat&);

//! This is the form of a non-const sparse Gradient() method.
template<typename FunctionType>
using SparseGradientForm = void(FunctionType::*)(
    const arma::mat&, const size_t, arma::sp_mat&, const size_t);

//! This is the form of a const sparse Gradient() method.
template<typename FunctionType>
using SparseGradientConstForm = void(FunctionType::*)(
    const arma::mat&, const size_t, arma::sp_mat&, const size_t) const;

//! This is the form of a static sparse Gradient() method.
template<typename FunctionType>
using SparseGradientStaticForm = void(*)(
    const arma::mat&, const size_t, arma::sp_mat&, const size_t);

//! This is the form of a non-const NumFeatures() method.
template<typename FunctionType>
using NumFeaturesForm = size_t(FunctionType::*)();

//! This is the form of a const NumFeatures() method.
template<typename FunctionType>
using NumFeaturesConstForm = size_t(FunctionType::*)() const;

//! This is the form of a static NumFeatures() method.
template<typename FunctionType>
using NumFeaturesStaticForm = size_t(*)();

//! This is the form of a non-const PartialGradient() method.
template<typename FunctionType>
using PartialGradientForm = void(FunctionType::*)(
    const arma::mat&, const size_t, arma::sp_mat&);

//! This is the form of a const PartialGradient() method.
template<typename FunctionType>
using PartialGradientConstForm = void(FunctionType::*)(
    const arma::mat&, const size_t, arma::sp_mat&) const;

//! This is the form of a static PartialGradient() method.
template<typename FunctionType>
using PartialGradientStaticForm = void(*)(
    const arma::mat&, const size_t, arma::sp_mat&);

//! This is a utility struct that will match any non-const form.
template<typename FunctionType, typename... Ts>
using OtherForm = double(FunctionType::*)(Ts...);

//! This is a utility struct that will match any const form.
template<typename FunctionType, typename... Ts>
using OtherConstForm = double(FunctionType::*)(Ts...) const;

//! This is a utility struct that will match any static form.
template<typename FunctionType, typename... Ts>
using OtherStaticForm = double(*)(Ts...);

/**
 * This is a utility type used to provide unusable overloads from each of the
 * mixin classes.  If you are seeing an error mentioning this class, the most
 * likely issue is that you have not implemented the right methods for your
 * FunctionType class.
 */
struct UnconstructableType
{
 private:
  UnconstructableType() { }
};

/**
 * Utility struct: sometimes we want to know if we have two functions available,
 * and that at least one of them is non-const and non-static.  If the
 * corresponding checkers (from HAS_METHOD_FORM()) are given as CheckerA and
 * CheckerB, and the corresponding non-const, const, and static function
 * signatures are given as SignatureA, ConstSignatureA, StaticSignatureA,
 * SignatureB, ConstSignatureB, and StaticSignatureB, then 'value' will be true
 * if methods with the correct names exist in the given ClassType and at least
 * one of those two methods is non-const and non-static.
 */
template<typename ClassType,
         template<typename, template<typename...> class, size_t> class CheckerA,
         template<typename...> class SignatureA,
         template<typename...> class ConstSignatureA,
         template<typename...> class StaticSignatureA,
         template<typename, template<typename...> class, size_t> class CheckerB,
         template<typename...> class SignatureB,
         template<typename...> class ConstSignatureB,
         template<typename...> class StaticSignatureB>
struct HasNonConstSignatures
{
  // Check if any const or static version of method A exists.
  const static bool HasAnyFormA =
      CheckerA<ClassType, SignatureA, 0>::value ||
      CheckerA<ClassType, ConstSignatureA, 0>::value ||
      CheckerA<ClassType, StaticSignatureA, 0>::value;
  // Check if any const or static versino of method B exists.
  const static bool HasAnyFormB =
      CheckerB<ClassType, SignatureB, 0>::value ||
      CheckerB<ClassType, ConstSignatureB, 0>::value ||
      CheckerB<ClassType, StaticSignatureB, 0>::value;

  // Make sure at least one const version exists.
  const static bool HasEitherNonConstForm =
      CheckerA<ClassType, SignatureA, 0>::value ||
      CheckerB<ClassType, SignatureB, 0>::value;

  const static bool value = HasEitherNonConstForm && HasAnyFormA && HasAnyFormB;
};

/**
 * Utility struct: sometimes we want to know if we have two functions available,
 * and that at least one of them is const and both of them are not non-const and
 * non-static.  If the corresponding checkers (from HAS_METHOD_FORM()) are given
 * as CheckerA and CheckerB, and the corresponding const and static function
 * signatures are given as ConstSignatureA, StaticSignatureA, ConstSignatureB,
 * and StaticSignatureB, then 'value' will be true if methods with the correct
 * names exist in the given ClassType and at least one of those two methods is
 * const, and neither method is non-const and non-static.
 */
template<typename ClassType,
         template<typename, template<typename...> class, size_t> class CheckerA,
         template<typename...> class ConstSignatureA,
         template<typename...> class StaticSignatureA,
         template<typename, template<typename...> class, size_t> class CheckerB,
         template<typename...> class ConstSignatureB,
         template<typename...> class StaticSignatureB>
struct HasConstSignatures
{
  // Check if any const or static version of method A exists.
  const static bool HasAnyFormA =
      CheckerA<ClassType, ConstSignatureA, 0>::value ||
      CheckerA<ClassType, StaticSignatureA, 0>::value;
  // Check if any const or static version of method B exists.
  const static bool HasAnyFormB =
      CheckerB<ClassType, ConstSignatureB, 0>::value ||
      CheckerB<ClassType, StaticSignatureB, 0>::value;

  // Make sure at least one const version exists.
  const static bool HasEitherConstForm =
      CheckerA<ClassType, ConstSignatureA, 0>::value ||
      CheckerB<ClassType, ConstSignatureB, 0>::value;

  const static bool value = HasEitherConstForm && HasAnyFormA && HasAnyFormB;
};

} // namespace traits
} // namespace optimization
} // namespace mlpack

#endif
