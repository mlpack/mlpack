/**
 * @file add_decomposable_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * Add decomposable variants of Evaluate(), Gradient(), and
 * EvaluateWithGradient().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADD_DECOMPOSABLE_EVALUATE_WITH_GRADIENT_CONST_HPP
#define MLPACK_CORE_OPTIMIZERS_ADD_DECOMPOSABLE_EVALUATE_WITH_GRADIENT_CONST_HPP

#include <mlpack/prereqs.hpp>
#include "traits.hpp"

namespace mlpack {
namespace optimization {

namespace aux {

template<typename FunctionType>
using DecomposableEvaluateForm = double(FunctionType::*)(const arma::mat&,
                                                         const size_t,
                                                         const size_t);

template<typename FunctionType>
using DecomposableEvaluateConstForm =
    double(FunctionType::*)(const arma::mat&, const size_t, const size_t) const;

template<typename FunctionType>
using DecomposableEvaluateStaticForm = double(*)(const arma::mat&,
                                                 size_t,
                                                 size_t);

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

} // namespace aux

/**
 * The AddDecomposableEvaluateWithGradient mixin class will provide a
 * decomposable Evaluate() and Gradient() method if the given class has a
 * decomposable EvaluateWithGradient() method, or it will provide a decomposable
 * EvaluateWithGradient() method if the class has a decomposable Evaluate() and
 * Gradient() method, or it will provide nothing in any other case.
 */
template<typename FunctionType,
         bool HasDecomposableEvaluate =
             aux::HasEvaluate<FunctionType,
                 aux::DecomposableEvaluateForm>::value ||
             aux::HasEvaluate<FunctionType,
                 aux::DecomposableEvaluateConstForm>::value ||
             aux::HasEvaluate<FunctionType,
                 aux::DecomposableEvaluateStaticForm>::value,
         bool HasGradient =
             aux::HasGradient<FunctionType,
                 aux::DecomposableGradientForm>::value ||
             aux::HasGradient<FunctionType,
                 aux::DecomposableGradientConstForm>::value ||
             aux::HasGradient<FunctionType,
                 aux::DecomposableGradientStaticForm>::value,
         bool HasEvaluateWithGradient =
             aux::HasEvaluateWithGradient<FunctionType,
                 aux::DecomposableEvaluateWithGradientForm>::value ||
             aux::HasEvaluateWithGradient<FunctionType,
                 aux::DecomposableEvaluateWithGradientConstForm>::value ||
             aux::HasEvaluateWithGradient<FunctionType,
                 aux::DecomposableEvaluateWithGradientStaticForm>::value>
class AddDecomposableEvaluateWithGradient : public FunctionType { };

/**
 * If the FunctionType has Evaluate() and Gradient() but not
 * EvaluateWithGradient(), we will provide the latter.
 */
template<typename FunctionType>
class AddDecomposableEvaluateWithGradient<FunctionType, true, true, false> :
    public FunctionType
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
                              const size_t begin,
                              arma::mat& gradient,
                              const size_t batchSize)
  {
    const double objective = FunctionType::Evaluate(coordinates, begin,
        batchSize);
    FunctionType::Gradient(coordinates, begin, gradient, batchSize);
    return objective;
  }
};

/**
 * If the FunctionType has EvaluateWithGradient() but not Evaluate(), provide
 * that function.
 */
template<typename FunctionType>
class AddDecomposableEvaluateWithGradient<FunctionType, false, true, true> :
    public FunctionType
{
 public:
  /**
   * Return the objective function for the given coordinates.
   *
   * @param coordinates Coordinates to evaluate the function at.
   */
  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    arma::mat gradient; // This will be ignored.
    return FunctionType::EvaluateWithGradient(coordinates, begin, gradient,
        batchSize);
  }
};

/**
 * If the FunctionType has EvaluateWithGradient() but not Gradient(), provide
 * that function.
 */
template<typename FunctionType>
class AddDecomposableEvaluateWithGradient<FunctionType, true, false, true> :
      public FunctionType
{
 public:
  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  void Gradient(const arma::mat& coordinates,
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
