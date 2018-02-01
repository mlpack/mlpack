/**
 * @file traits.hpp
 * @author Ryan Curtin
 *
 * This file provides metaprogramming utilities for detecting certain members of
 * FunctionType classes.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_TRAITS_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_TRAITS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {
namespace optimization {
namespace traits {

HAS_EXACT_METHOD_FORM(Evaluate, HasEvaluate);
HAS_EXACT_METHOD_FORM(Gradient, HasGradient);
HAS_EXACT_METHOD_FORM(EvaluateWithGradient, HasEvaluateWithGradient);

template<typename FunctionType>
using EvaluateForm = double(FunctionType::*)(const arma::mat&);

template<typename FunctionType>
using EvaluateConstForm =
    double(FunctionType::*)(const arma::mat&) const;

template<typename FunctionType>
using EvaluateStaticForm = double(*)(const arma::mat&);

template<typename FunctionType>
using GradientForm = void(FunctionType::*)(const arma::mat&, arma::mat&);

template<typename FunctionType>
using GradientConstForm =
    void(FunctionType::*)(const arma::mat&, arma::mat&) const;

template<typename FunctionType>
using GradientStaticForm = void(*)(const arma::mat&, arma::mat&);

template<typename FunctionType>
using EvaluateWithGradientForm =
    double(FunctionType::*)(const arma::mat&, arma::mat&);

template<typename FunctionType>
using EvaluateWithGradientConstForm =
    double(FunctionType::*)(const arma::mat&, arma::mat&) const;

template<typename FunctionType>
using EvaluateWithGradientStaticForm =
    double(*)(const arma::mat&, arma::mat&);

template<typename FunctionType>
using DecomposableEvaluateForm = double(FunctionType::*)(const arma::mat&,
                                                         const size_t,
                                                         const size_t);

template<typename FunctionType>
using DecomposableEvaluateConstForm =
    double(FunctionType::*)(const arma::mat&, const size_t, const size_t) const;

template<typename FunctionType>
using DecomposableEvaluateStaticForm = double(*)(const arma::mat&,
                                                 const size_t,
                                                 const size_t);

template<typename FunctionType>
using DecomposableGradientForm = void(FunctionType::*)(const arma::mat&,
                                                       const size_t,
                                                       arma::mat&,
                                                       const size_t);

template<typename FunctionType>
using DecomposableGradientConstForm =
    void(FunctionType::*)(const arma::mat&,
                          const size_t,
                          arma::mat&,
                          const size_t) const;

template<typename FunctionType, typename... Ts>
using DecomposableGradientStaticForm = void(*)(const arma::mat&,
                                               const size_t,
                                               arma::mat&);

template<typename FunctionType>
using DecomposableEvaluateWithGradientForm =
    double(FunctionType::*)(const arma::mat&,
                            const size_t,
                            arma::mat&,
                            const size_t);

template<typename FunctionType>
using DecomposableEvaluateWithGradientConstForm =
    void(FunctionType::*)(const arma::mat&,
                          const size_t,
                          arma::mat&,
                          const size_t) const;

template<typename FunctionType>
using DecomposableEvaluateWithGradientStaticForm =
    double(*)(const arma::mat&, const size_t, arma::mat&, const size_t);

HAS_METHOD_FORM(Evaluate, HasOtherEvaluate);
HAS_METHOD_FORM(Gradient, HasOtherGradient);
HAS_METHOD_FORM(EvaluateWithGradient, HasOtherEvaluateWithGradient);

template<typename FunctionType, typename... Ts>
using OtherForm = double(FunctionType::*)(Ts...);

template<typename FunctionType, typename... Ts>
using OtherConstForm = double(FunctionType::*)(Ts...) const;

template<typename FunctionType, typename... Ts>
using OtherStaticForm = double(*)(Ts...);

struct UnconstructableType
{
 private:
  UnconstructableType() { }
};

} // namespace traits
} // namespace optimization
} // namespace mlpack

#endif
