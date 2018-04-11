/**
 * @file add_decomposable_evaluate.hpp
 * @author Ryan Curtin
 *
 * Adds a decomposable Evaluate() function if a decomposable
 * EvaluateWithGradient() function exists.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_DECOMPOSABLE_EVALUATE_HPP

#include "traits.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AddDecomposableEvaluate mixin class will add a decomposable Evaluate()
 * method if a decomposable EvaluateWithGradient() function exists, or nothing
 * otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientForm>::value,
         bool HasDecomposableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::DecomposableEvaluateForm>::value>
class AddDecomposableEvaluate
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  double Evaluate(traits::UnconstructableType&, const size_t, const size_t);
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType, bool HasDecomposableEvaluateWithGradient>
class AddDecomposableEvaluate<FunctionType, HasDecomposableEvaluateWithGradient,
    true>
{
 public:
  // Reflect the existing Evaluate().
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    return static_cast<FunctionType*>(
        static_cast<Function<FunctionType>*>(this))->Evaluate(coordinates,
        begin, batchSize);
  }
};

/**
 * If we have a decomposable EvaluateWithGradient() but not a decomposable
 * Evaluate(), add a decomposable Evaluate() method.
 */
template<typename FunctionType>
class AddDecomposableEvaluate<FunctionType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   */
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    arma::mat gradient; // This will be ignored.
    return static_cast<Function<FunctionType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddDecomposableEvaluateConst mixin class will add a decomposable const
 * Evaluate() method if a decomposable const EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientConstForm>::value,
         bool HasDecomposableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::DecomposableEvaluateConstForm>::value>
class AddDecomposableEvaluateConst
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  double Evaluate(traits::UnconstructableType&, const size_t, const size_t)
      const;
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType, bool HasDecomposableEvaluateWithGradient>
class AddDecomposableEvaluateConst<FunctionType,
    HasDecomposableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize) const
  {
    return static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType>*>(this))->Evaluate(coordinates,
        begin, batchSize);
  }
};

/**
 * If we have a decomposable const EvaluateWithGradient() but not a decomposable
 * const Evaluate(), add a decomposable const Evaluate() method.
 */
template<typename FunctionType>
class AddDecomposableEvaluateConst<FunctionType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   */
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize) const
  {
    arma::mat gradient; // This will be ignored.
    return
        static_cast<const Function<FunctionType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddDecomposableEvaluateStatic mixin class will add a decomposable static
 * Evaluate() method if a decomposable static EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientStaticForm>::value,
         bool HasDecomposableEvaluate =
             traits::HasEvaluate<FunctionType,
                 traits::DecomposableEvaluateStaticForm>::value>
class AddDecomposableEvaluateStatic
{
 public:
  // Provide a dummy overload so the name 'Evaluate' exists for this object.
  double Evaluate(traits::UnconstructableType&, const size_t) const;
};

/**
 * Reflect the existing Evaluate().
 */
template<typename FunctionType, bool HasDecomposableEvaluateWithGradient>
class AddDecomposableEvaluateStatic<FunctionType,
    HasDecomposableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Evaluate().
  static double Evaluate(const arma::mat& coordinates,
                         const size_t begin,
                         const size_t batchSize)
  {
    return FunctionType::Evaluate(coordinates, begin, batchSize);
  }
};

/**
 * If we have a decomposable EvaluateWithGradient() but not a decomposable
 * Evaluate(), add a decomposable Evaluate() method.
 */
template<typename FunctionType>
class AddDecomposableEvaluateStatic<FunctionType, true, false>
{
 public:
  /**
   * Return the objective function for the given coordinates, starting at the
   * given decomposable function using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of first function to evaluate.
   * @param batchSize Number of functions to evaluate.
   */
  static double Evaluate(const arma::mat& coordinates,
                         const size_t begin,
                         const size_t batchSize)
  {
    arma::mat gradient; // This will be ignored.
    return FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
