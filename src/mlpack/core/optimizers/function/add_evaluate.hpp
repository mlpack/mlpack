/**
 * @file add_evaluate.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that the
 * function Evaluate() is avaliable if EvaluateWithGradient() is available.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_EVALUATE_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_EVALUATE_HPP

#include <mlpack/prereqs.hpp>
#include "traits.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AddEvaluate mixin class will provide an Evaluate() method if the given
 * FunctionType has EvaluateWithGradient(), or nothing otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientForm>::value,
         bool HasEvaluate =
             traits::HasEvaluate<FunctionType, traits::EvaluateForm>::value>
class AddEvaluate
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  double Evaluate(traits::UnconstructableType&);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType, bool HasEvaluateWithGradient>
class AddEvaluate<FunctionType, HasEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  double Evaluate(const arma::mat& coordinates)
  {
    return static_cast<FunctionType*>(static_cast<Function<FunctionType>*>(
        this))->Evaluate(coordinates);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Evaluate(), add an
 * Evaluate() method.
 */
template<typename FunctionType>
class AddEvaluate<FunctionType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  double Evaluate(const arma::mat& coordinates)
  {
    arma::mat gradient; // This will be ignored.
    return static_cast<Function<FunctionType>*>(this)->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * The AddEvaluateConst mixin class will provide a const Evaluate() method if
 * the given FunctionType has EvaluateWithGradient() const, or nothing
 * otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientConstForm>::value,
         bool HasEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::EvaluateConstForm>::value>
class AddEvaluateConst
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  double Evaluate(traits::UnconstructableType&) const;
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType, bool HasEvaluateWithGradient>
class AddEvaluateConst<FunctionType, HasEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  double Evaluate(const arma::mat& coordinates) const
  {
    return static_cast<const FunctionType*>(static_cast<const
Function<FunctionType>*>(this))->Evaluate(
        coordinates);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Evaluate(), add an
 * Evaluate() without a using directive to make the base Evaluate() accessible.
 */
template<typename FunctionType>
class AddEvaluateConst<FunctionType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  double Evaluate(const arma::mat& coordinates) const
  {
    arma::mat gradient; // This will be ignored.
    return static_cast<
        const Function<FunctionType>*>(this)->EvaluateWithGradient(coordinates,
        gradient);
  }
};

/**
 * The AddEvaluateStatic mixin class will provide a static Evaluate() method if
 * the given FunctionType has EvaluateWithGradient() static, or nothing
 * otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientStaticForm>::value,
         bool HasEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::EvaluateStaticForm>::value>
class AddEvaluateStatic
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  static double Evaluate(traits::UnconstructableType&);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType, bool HasEvaluateWithGradient>
class AddEvaluateStatic<FunctionType, HasEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  static double Evaluate(const arma::mat& coordinates)
  {
    return FunctionType::Evaluate(coordinates);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Evaluate(), add an
 * Evaluate() without a using directive to make the base Evaluate() accessible.
 */
template<typename FunctionType>
class AddEvaluateStatic<FunctionType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  static double Evaluate(const arma::mat& coordinates)
  {
    arma::mat gradient; // This will be ignored.
    return FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
