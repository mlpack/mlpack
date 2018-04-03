/**
 * @file add_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that the
 * EvaluateWithGradient() function is available if possible.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_EVALUATE_WITH_GRADIENT_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_EVALUATE_WITH_GRADIENT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>
#include "traits.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AddEvaluateWithGradient mixin class will provide an
 * EvaluateWithGradient() method if the given FunctionType has both Evaluate()
 * and Gradient(), or it will provide nothing otherwise.
 */
template<typename FunctionType,
         // Check if there is at least one non-const Evaluate() or Gradient().
         bool HasEvaluateGradient = traits::HasNonConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::EvaluateForm,
             traits::EvaluateConstForm,
             traits::EvaluateStaticForm,
             traits::HasGradient,
             traits::GradientForm,
             traits::GradientConstForm,
             traits::GradientStaticForm>::value,
         bool HasEvaluateWithGradient = traits::HasEvaluateWithGradient<
             FunctionType,
             traits::EvaluateWithGradientForm>::value>
class AddEvaluateWithGradient
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  double EvaluateWithGradient(traits::UnconstructableType&);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType, bool HasEvaluateGradient>
class AddEvaluateWithGradient<FunctionType, HasEvaluateGradient, true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType>*>(this))->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * If the FunctionType has Evaluate() and Gradient(), provide
 * EvaluateWithGradient().
 */
template<typename FunctionType>
class AddEvaluateWithGradient<FunctionType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  double EvaluateWithGradient(const arma::mat& coordinates,
                              arma::mat& gradient)
  {
    const double objective =
        static_cast<Function<FunctionType>*>(this)->Evaluate(coordinates);
    static_cast<Function<FunctionType>*>(this)->Gradient(coordinates, gradient);
    return objective;
  }
};

/**
 * The AddEvaluateWithGradient mixin class will provide an
 * EvaluateWithGradient() const method if the given FunctionType has both
 * Evaluate() const and Gradient() const, or it will provide nothing otherwise.
 */
template<typename FunctionType,
         // Check if there is at least one const Evaluate() or Gradient().
         bool HasEvaluateGradient = traits::HasConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::EvaluateConstForm,
             traits::EvaluateStaticForm,
             traits::HasGradient,
             traits::GradientConstForm,
             traits::GradientStaticForm>::value,
         bool HasEvaluateWithGradient = traits::HasEvaluateWithGradient<
             FunctionType,
             traits::EvaluateWithGradientConstForm>::value>
class AddEvaluateWithGradientConst
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  double EvaluateWithGradient(traits::UnconstructableType&) const;
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType, bool HasEvaluateGradient>
class AddEvaluateWithGradientConst<FunctionType, HasEvaluateGradient, true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
      const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType>*>(this))->EvaluateWithGradient(
        coordinates, gradient);
  }
};

/**
 * If the FunctionType has Evaluate() const and Gradient() const, provide
 * EvaluateWithGradient() const.
 */
template<typename FunctionType>
class AddEvaluateWithGradientConst<FunctionType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  double EvaluateWithGradient(const arma::mat& coordinates,
                              arma::mat& gradient) const
  {
    const double objective =
        static_cast<const Function<FunctionType>*>(this)->Evaluate(coordinates);
    static_cast<const Function<FunctionType>*>(this)->Gradient(coordinates,
        gradient);
    return objective;
  }
};

/**
 * The AddEvaluateWithGradientStatic mixin class will provide a
 * static EvaluateWithGradient() method if the given FunctionType has both
 * static Evaluate() and static Gradient(), or it will provide nothing
 * otherwise.
 */
template<typename FunctionType,
         bool HasEvaluateGradient =
             traits::HasEvaluate<FunctionType,
                 traits::EvaluateStaticForm>::value &&
             traits::HasGradient<FunctionType,
                 traits::GradientStaticForm>::value,
         bool HasEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::EvaluateWithGradientStaticForm>::value>
class AddEvaluateWithGradientStatic
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  static double EvaluateWithGradient(traits::UnconstructableType&);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType, bool HasEvaluateGradient>
class AddEvaluateWithGradientStatic<FunctionType, HasEvaluateGradient, true>
{
 public:
  // Reflect the existing EvaluateWithGradient().
  static double EvaluateWithGradient(const arma::mat& coordinates,
                                     arma::mat& gradient)
  {
    return FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

/**
 * If the FunctionType has static Evaluate() and static Gradient(), provide
 * static EvaluateWithGradient().
 */
template<typename FunctionType>
class AddEvaluateWithGradientStatic<FunctionType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  static double EvaluateWithGradient(const arma::mat& coordinates,
                                     arma::mat& gradient)
  {
    const double objective = FunctionType::Evaluate(coordinates);
    FunctionType::Gradient(coordinates, gradient);
    return objective;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
