/**
 * @file add_evaluate_with_gradient.hpp
 * @author Ryan Curtin
 *
 * This file defines a mixin for the Function class that will ensure that all
 * three of the functions Evaluate(), Gradient(), and EvaluateWithGradient() are
 * available, if some are.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_ADD_EVALUATE_WITH_GRADIENT_HPP
#define MLPACK_CORE_OPTIMIZERS_ADD_EVALUATE_WITH_GRADIENT_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {
namespace optimization {

namespace aux {

HAS_METHOD_FORM(Evaluate, HasEvaluate);
HAS_METHOD_FORM(Gradient, HasGradient);
HAS_METHOD_FORM(EvaluateWithGradient, HasEvaluateWithGradient);

template<typename FunctionType, typename... Ts>
using EvaluateForm = double(FunctionType::*)(const arma::mat&, Ts...);

template<typename FunctionType, typename... Ts>
using EvaluateConstForm =
    double(FunctionType::*)(const arma::mat&, Ts...) const;

template<typename FunctionType, typename... Ts>
using EvaluateStaticForm = double(*)(const arma::mat&, Ts...);

template<typename FunctionType, typename... Ts>
using GradientForm = void(FunctionType::*)(const arma::mat&, arma::mat&, Ts...);

template<typename FunctionType, typename... Ts>
using GradientConstForm =
    void(FunctionType::*)(const arma::mat&, arma::mat&, Ts...) const;

template<typename FunctionType, typename... Ts>
using GradientStaticForm = void(*)(const arma::mat&, arma::mat&, Ts...);

template<typename FunctionType, typename... Ts>
using EvaluateWithGradientForm =
    double(FunctionType::*)(const arma::mat&, arma::mat&, Ts...);

template<typename FunctionType, typename... Ts>
using EvaluateWithGradientConstForm =
    void(FunctionType::*)(const arma::mat&, arma::mat&, Ts...) const;

template<typename FunctionType, typename... Ts>
using EvaluateWithGradientStaticForm =
    double(*)(const arma::mat&, arma::mat&, Ts...);

} // namespace aux

/**
 * The AddEvaluateWithGradient mixin class will provide an Evaluate() and
 * Gradient() method if the given FunctionType has EvaluateWithGradient(), or it
 * will provide an EvaluateWithGradient() method if the given FunctionType has
 * Evaluate() and Gradient(), or it will provide nothing in any other case.
 */
template<typename FunctionType,
         bool HasEvaluate =
             aux::HasEvaluate<FunctionType, aux::EvaluateForm>::value ||
             aux::HasEvaluate<FunctionType, aux::EvaluateConstForm>::value ||
             aux::HasEvaluate<FunctionType, aux::EvaluateStaticForm>::value,
         bool HasGradient =
             aux::HasGradient<FunctionType, aux::GradientForm>::value ||
             aux::HasGradient<FunctionType, aux::GradientConstForm>::value ||
             aux::HasGradient<FunctionType, aux::GradientStaticForm>::value,
         bool HasEvaluateWithGradient =
             aux::HasEvaluateWithGradient<FunctionType,
                 aux::EvaluateWithGradientForm>::value ||
             aux::HasEvaluateWithGradient<FunctionType,
                 aux::EvaluateWithGradientConstForm>::value ||
             aux::HasEvaluateWithGradient<FunctionType,
                 aux::EvaluateWithGradientStaticForm>::value>
class AddEvaluateWithGradient : public FunctionType { };

/**
 * If the FunctionType has Evaluate() and Gradient() but not
 * EvaluateWithGradient(), we will provide the latter.
 */
template<typename FunctionType>
class AddEvaluateWithGradient<FunctionType, true, true, false> :
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
                              arma::mat& gradient)
  {
    const double objective = FunctionType::Evaluate(coordinates);
    FunctionType::Gradient(coordinates, gradient);
    return objective;
  }
};

/**
 * If the FunctionType has EvaluateWithGradient() but not Evaluate(), provide
 * that function.
 */
template<typename FunctionType>
class AddEvaluateWithGradient<FunctionType, false, true, true> :
    public FunctionType
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
    return FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

/**
 * If the FunctionType has EvaluateWithGradient() but not Gradient(), provide
 * that function.
 */
template<typename FunctionType>
class AddEvaluateWithGradient<FunctionType, true, false, true> :
      public FunctionType
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
    (void) FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

/**
 * If the FunctionType has EvaluateWithGradient() but neither Evaluate() nor
 * Gradient(), provide both.  We can't inherit from two other overloads because
 * this causes FunctionType to be inherited multiple times.
 */
template<typename FunctionType>
class AddEvaluateWithGradient<FunctionType, false, false, true> :
    public FunctionType
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
    return FunctionType::EvaluateWithGradient(coordinates, gradient);
  }

  /**
   * Calculate the gradient and store it in the given matrix.
   *
   * @param coordinates Coordinates to evaluate the function at.
   * @param gradient Matrix to store the gradient into.
   */
  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    // The returned objective value will be ignored.
    (void) FunctionType::EvaluateWithGradient(coordinates, gradient);
  }
};

} // namespace optimization
} // namespace mlpack

#endif
