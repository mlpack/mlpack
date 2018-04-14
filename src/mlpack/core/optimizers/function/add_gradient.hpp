/**
 * @file add_gradient.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that the
 * function Gradient() is avaiable if EvaluateWithGradient() is available.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_GRADIENT_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_GRADIENT_HPP

#include <mlpack/prereqs.hpp>
#include "traits.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AddGradient mixin class will provide a Gradient() method if the given
 * FunctionType has EvaluateWithGradient(), or nothing otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientForm>::value,
         bool HasGradient = traits::HasGradient<FunctionType,
             traits::GradientForm>::value>
class AddGradient
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&) { }
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType, bool HasEvaluateWithGradient>
class AddGradient<FunctionType, HasEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    static_cast<FunctionType*>(static_cast<Function<FunctionType>*>(
        this))->Gradient(coordinates, gradient);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Gradient(), add an
 * Gradient() without a using directive to make the base Gradient() accessible.
 */
template<typename FunctionType>
class AddGradient<FunctionType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    // The returned objective value will be ignored.
    (void) static_cast<Function<FunctionType>*>(this)->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * The AddGradient mixin class will provide a const Gradient() method if the
 * given FunctionType has EvaluateWithGradient() const, or nothing otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientConstForm>::value,
         bool HasGradient = traits::HasGradient<FunctionType,
             traits::GradientConstForm>::value>
class AddGradientConst
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&) const { }
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType, bool HasEvaluateWithGradient>
class AddGradientConst<FunctionType, HasEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const
  {
    static_cast<const FunctionType*>(static_cast<const
Function<FunctionType>*>(this))->Gradient(coordinates,
        gradient);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Gradient(), add a
 * Gradient() without a using directive to make the base Gradient() accessible.
 */
template<typename FunctionType>
class AddGradientConst<FunctionType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient) const
  {
    // The returned objective value will be ignored.
    (void) static_cast<
        const Function<FunctionType>*>(this)->EvaluateWithGradient(coordinates,
        gradient);
  }
};

/**
 * The AddGradient mixin class will provide a static Gradient() method if the
 * given FunctionType has static EvaluateWithGradient(), or nothing otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientStaticForm>::value,
         bool HasGradient = traits::HasGradient<FunctionType,
             traits::GradientStaticForm>::value>
class AddGradientStatic
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  static void Gradient(traits::UnconstructableType&) { }
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType, bool HasEvaluateWithGradient>
class AddGradientStatic<FunctionType, HasEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  static void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    FunctionType::Gradient(coordinates, gradient);
  }
};

/**
 * If we have EvaluateWithGradient() but no existing Gradient(), add a
 * Gradient() without a using directive to make the base Gradient() accessible.
 */
template<typename FunctionType>
class AddGradientStatic<FunctionType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  static void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    // The returned objective value will be ignored.
    (void) FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
