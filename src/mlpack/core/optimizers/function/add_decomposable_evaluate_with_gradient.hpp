/**
 * @file add_decomposable_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * Adds a decomposable EvaluateWithGradient() function if both a decomposable
 * Evaluate() and a decomposable Gradient() function exist.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_W_GRADIENT_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_W_GRADIENT_HPP

#include "traits.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AddDecomposableEvaluateWithGradient mixin class will add a decomposable
 * EvaluateWithGradient() method if a decomposable Evaluate() method and a
 * decomposable Gradient() method exists, or nothing otherwise.
 */
template<typename FunctionType,
         // Check if there is at least one non-const Evaluate() or Gradient().
         bool HasDecomposableEvaluateGradient = traits::HasNonConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::DecomposableEvaluateForm,
             traits::DecomposableEvaluateConstForm,
             traits::DecomposableEvaluateStaticForm,
             traits::HasGradient,
             traits::DecomposableGradientForm,
             traits::DecomposableGradientConstForm,
             traits::DecomposableGradientStaticForm>::value,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientForm>::value>
class AddDecomposableEvaluateWithGradient
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  double EvaluateWithGradient(traits::UnconstructableType&, const size_t,
      const size_t);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType, bool HasDecomposableEvaluateGradient>
class AddDecomposableEvaluateWithGradient<FunctionType,
    HasDecomposableEvaluateGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType>*>(this))->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have a both decomposable Evaluate() and a decomposable Gradient() but
 * not a decomposable EvaluateWithGradient(), add a decomposable
 * EvaluateWithGradient() method.
 */
template<typename FunctionType>
class AddDecomposableEvaluateWithGradient<FunctionType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   */
  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize)
  {
    const double objective =
        static_cast<Function<FunctionType>*>(this)->Evaluate(coordinates, begin,
        batchSize);
    static_cast<Function<FunctionType>*>(this)->Gradient(coordinates, begin,
        gradient, batchSize);
    return objective;
  }
};

/**
 * The AddDecomposableEvaluateWithGradientConst mixin class will add a
 * decomposable const EvaluateWithGradient() method if both a decomposable const
 * Evaluate() and a decomposable const Gradient() function exist, or nothing
 * otherwise.
 */
template<typename FunctionType,
         // Check if there is at least one const Evaluate() or Gradient().
         bool HasDecomposableEvaluateGradient = traits::HasConstSignatures<
             FunctionType,
             traits::HasEvaluate,
             traits::DecomposableEvaluateConstForm,
             traits::DecomposableEvaluateStaticForm,
             traits::HasGradient,
             traits::DecomposableGradientConstForm,
             traits::DecomposableGradientStaticForm>::value,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientConstForm>::value>
class AddDecomposableEvaluateWithGradientConst
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  double EvaluateWithGradient(traits::UnconstructableType&, const size_t,
      const size_t) const;
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType, bool HasDecomposableEvaluateGradient>
class AddDecomposableEvaluateWithGradientConst<FunctionType,
    HasDecomposableEvaluateGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize) const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType>*>(this))->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have both a decomposable const Evaluate() and a decomposable const
 * Gradient() but not a decomposable const EvaluateWithGradient(), add a
 * decomposable const EvaluateWithGradient() method.
 */
template<typename FunctionType>
class AddDecomposableEvaluateWithGradientConst<FunctionType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   */
  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize) const
  {
    const double objective =
        static_cast<const Function<FunctionType>*>(this)->Evaluate(coordinates,
        begin, batchSize);
    static_cast<const Function<FunctionType>*>(this)->Gradient(coordinates,
        begin, gradient, batchSize);
    return objective;
  }
};

/**
 * The AddDecomposableEvaluateWithGradientStatic mixin class will add a
 * decomposable static EvaluateWithGradient() method if both a decomposable
 * static Evaluate() and a decomposable static gradient() function exist, or
 * nothing otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateGradient =
             traits::HasEvaluate<FunctionType,
                 traits::DecomposableEvaluateStaticForm>::value &&
             traits::HasGradient<FunctionType,
                 traits::DecomposableGradientStaticForm>::value,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientStaticForm>::value>
class AddDecomposableEvaluateWithGradientStatic
{
 public:
  // Provide a dummy overload so the name 'EvaluateWithGradient' exists for this
  // object.
  static double EvaluateWithGradient(traits::UnconstructableType&, const size_t,
      const size_t);
};

/**
 * Reflect the existing EvaluateWithGradient().
 */
template<typename FunctionType, bool HasDecomposableEvaluateGradient>
class AddDecomposableEvaluateWithGradientStatic<FunctionType,
    HasDecomposableEvaluateGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  static double EvaluateWithGradient(const arma::mat& coordinates,
                                     const size_t begin,
                                     arma::mat& gradient,
                                     const size_t batchSize)
  {
    return FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

/**
 * If we have a decomposable static Evaluate() and a decomposable static
 * Gradient() but not a decomposable static EvaluateWithGradient(), add a
 * decomposable static Gradient() method.
 */
template<typename FunctionType>
class AddDecomposableEvaluateWithGradientStatic<FunctionType, true, false>
{
 public:
  /**
   * Return both the evaluated objective function and its gradient, storing the
   * gradient in the given matrix, starting at the given decomposable function
   * and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to begin with.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to evaluate.
   */
  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize) const
  {
    const double objective = FunctionType::Evaluate(coordinates, begin,
        batchSize);
    FunctionType::Gradient(coordinates, begin, gradient, batchSize);
    return objective;
  }
};

} // namespace optimization
} // namespace mlpack

#endif
