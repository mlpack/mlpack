/**
 * @file add_decomposable_gradient.hpp
 * @author Ryan Curtin
 *
 * Adds a decomposable Gradient() function if a decomposable
 * EvaluateWithGradient() function exists.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_DECOMPOSABLE_GRADIENT_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_ADD_DECOMPOSABLE_GRADIENT_HPP

#include "traits.hpp"

namespace mlpack {
namespace optimization {

/**
 * The AddDecomposableGradient mixin class will add a decomposable Gradient()
 * method if a decomposable EvaluateWithGradient() function exists, or nothing
 * otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientForm>::value,
         bool HasDecomposableGradient =
             traits::HasGradient<FunctionType,
                 traits::DecomposableGradientForm>::value>
class AddDecomposableGradient
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&, const size_t, const size_t);
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType, bool HasDecomposableEvaluateWithGradient>
class AddDecomposableGradient<FunctionType, HasDecomposableEvaluateWithGradient,
    true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize)
  {
    static_cast<FunctionType*>(
        static_cast<Function<FunctionType>*>(this))->Gradient(coordinates,
        begin, gradient, batchSize);
  }
};

/**
 * If we have a decomposable EvaluateWithGradient() but not a decomposable
 * Gradient(), add a decomposable Evaluate() method.
 */
template<typename FunctionType>
class AddDecomposableGradient<FunctionType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given decomposable function index and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to calculate for.
   */
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize)
  {
    // The returned objective value will be ignored.
    (void) static_cast<Function<FunctionType>*>(this)->EvaluateWithGradient(
        coordinates, begin, gradient, batchSize);
  }
};

/**
 * The AddDecomposableGradientConst mixin class will add a decomposable const
 * Gradient() method if a decomposable const EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientConstForm>::value,
         bool HasDecomposableGradient =
             traits::HasGradient<FunctionType,
                 traits::DecomposableGradientConstForm>::value>
class AddDecomposableGradientConst
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  void Gradient(traits::UnconstructableType&, const size_t, const size_t) const;
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType, bool HasDecomposableEvaluateWithGradient>
class AddDecomposableGradientConst<FunctionType,
    HasDecomposableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize) const
  {
    static_cast<const FunctionType*>(
        static_cast<const Function<FunctionType>*>(this))->Gradient(coordinates,
        begin, gradient, batchSize);
  }
};

/**
 * If we have a decomposable const EvaluateWithGradient() but not a decomposable
 * const Gradient(), add a decomposable const Gradient() method.
 */
template<typename FunctionType>
class AddDecomposableGradientConst<FunctionType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given decomposable function index and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to calculate for.
   */
  void Gradient(const arma::mat& coordinates,
                const size_t begin,
                arma::mat& gradient,
                const size_t batchSize) const
  {
    // The returned objective value will be ignored.
    (void) static_cast<
        const Function<FunctionType>*>(this)->EvaluateWithGradient(coordinates,
        begin, gradient, batchSize);
  }
};

/**
 * The AddDecomposableEvaluateStatic mixin class will add a decomposable static
 * Gradient() method if a decomposable static EvaluateWithGradient() function
 * exists, or nothing otherwise.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluateWithGradient =
             traits::HasEvaluateWithGradient<FunctionType,
                 traits::DecomposableEvaluateWithGradientStaticForm>::value,
         bool HasDecomposableGradient =
             traits::HasGradient<FunctionType,
                 traits::DecomposableGradientStaticForm>::value>
class AddDecomposableGradientStatic
{
 public:
  // Provide a dummy overload so the name 'Gradient' exists for this object.
  static void Gradient(traits::UnconstructableType&,
                       const size_t,
                       const size_t);
};

/**
 * Reflect the existing Gradient().
 */
template<typename FunctionType, bool HasDecomposableEvaluateWithGradient>
class AddDecomposableGradientStatic<FunctionType,
    HasDecomposableEvaluateWithGradient, true>
{
 public:
  // Reflect the existing Gradient().
  static void Gradient(const arma::mat& coordinates,
                       const size_t begin,
                       arma::mat& gradient,
                       const size_t batchSize)
  {
    FunctionType::Gradient(coordinates, begin, gradient, batchSize);
  }
};

/**
 * If we have a decomposable EvaluateWithGradient() but not a decomposable
 * Gradient(), add a decomposable Gradient() method.
 */
template<typename FunctionType>
class AddDecomposableGradientStatic<FunctionType, true, false>
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix, starting at the
   * given decomposable function index and using the given batch size.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param begin Index of decomposable function to start at.
   * @param gradient Matrix to store the gradient into.
   * @param batchSize Number of decomposable functions to calculate for.
   */
  static void Gradient(const arma::mat& coordinates,
                       const size_t begin,
                       arma::mat& gradient,
                       const size_t batchSize)
  {
    // The returned objective value will be ignored.
    (void) FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
