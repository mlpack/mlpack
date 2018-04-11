/**
 * @file function.hpp
 * @author Ryan Curtin
 *
 * The Function class is a wrapper class for any objective function that
 * provides any of the functions that an optimizer might use.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_FUNCTION_HPP
#define MLPACK_CORE_OPTIMIZERS_FUNCTION_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace optimization {

template<typename FunctionType>
class Function;

} // namespace optimization
} // namespace mlpack

#include "function/traits.hpp"
#include "function/static_checks.hpp"
#include "function/add_evaluate.hpp"
#include "function/add_gradient.hpp"
#include "function/add_evaluate_with_gradient.hpp"
#include "function/add_decomposable_evaluate.hpp"
#include "function/add_decomposable_gradient.hpp"
#include "function/add_decomposable_evaluate_with_gradient.hpp"

namespace mlpack {
namespace optimization {

/**
 * The Function class is a wrapper class for any FunctionType that will add any
 * possible derived methods.  For instance, if the given FunctionType has
 * Evaluate() and Gradient(), then Function<FunctionType> will have
 * EvaluateWithGradient().  This infrastructure allows two things:
 *
 *   1. Optimizers can expect FunctionTypes to have a wider array of functions
 *      than those FunctionTypes may actually implement.
 *
 *   2. FunctionTypes don't need to implement every single method that an
 *      optimizer might require, just those from which every method can be
 *      inferred.
 *
 * This class works by inheriting from a large set of "mixin" classes that
 * provide missing functions, if needed.  For instance, the AddGradient<> mixin
 * will provide a Gradient() method if the given FunctionType implements an
 * EvaluateWithGradient() method.
 *
 * Since all of the casting is static and each of the mixin classes is an empty
 * class, there should be no runtime overhead at all for this functionality.  In
 * addition, this class does not (to the best of my knowledge) rely on any
 * undefined behavior.
 */
template<typename FunctionType>
class Function :
    public AddDecomposableEvaluateWithGradientStatic<FunctionType>,
    public AddDecomposableEvaluateWithGradientConst<FunctionType>,
    public AddDecomposableEvaluateWithGradient<FunctionType>,
    public AddDecomposableGradientStatic<FunctionType>,
    public AddDecomposableGradientConst<FunctionType>,
    public AddDecomposableGradient<FunctionType>,
    public AddDecomposableEvaluateStatic<FunctionType>,
    public AddDecomposableEvaluateConst<FunctionType>,
    public AddDecomposableEvaluate<FunctionType>,
    public AddEvaluateWithGradientStatic<FunctionType>,
    public AddEvaluateWithGradientConst<FunctionType>,
    public AddEvaluateWithGradient<FunctionType>,
    public AddGradientStatic<FunctionType>,
    public AddGradientConst<FunctionType>,
    public AddGradient<FunctionType>,
    public AddEvaluateStatic<FunctionType>,
    public AddEvaluateConst<FunctionType>,
    public AddEvaluate<FunctionType>,
    public FunctionType
{
 public:
  // All of the mixin classes either reflect existing functionality or provide
  // an unconstructable overload with the same name, so we can use using
  // declarations here to ensure that they are all accessible.  Since we don't
  // know what FunctionType has, we can't use any using declarations there.
  using AddDecomposableEvaluateWithGradientStatic<
      FunctionType>::EvaluateWithGradient;
  using AddDecomposableEvaluateWithGradientConst<
      FunctionType>::EvaluateWithGradient;
  using AddDecomposableEvaluateWithGradient<FunctionType>::EvaluateWithGradient;
  using AddDecomposableGradientStatic<FunctionType>::Gradient;
  using AddDecomposableGradientConst<FunctionType>::Gradient;
  using AddDecomposableGradient<FunctionType>::Gradient;
  using AddDecomposableEvaluateStatic<FunctionType>::Evaluate;
  using AddDecomposableEvaluateConst<FunctionType>::Evaluate;
  using AddDecomposableEvaluate<FunctionType>::Evaluate;
  using AddEvaluateWithGradientStatic<FunctionType>::EvaluateWithGradient;
  using AddEvaluateWithGradientConst<FunctionType>::EvaluateWithGradient;
  using AddEvaluateWithGradient<FunctionType>::EvaluateWithGradient;
  using AddGradientStatic<FunctionType>::Gradient;
  using AddGradientConst<FunctionType>::Gradient;
  using AddGradient<FunctionType>::Gradient;
  using AddEvaluateStatic<FunctionType>::Evaluate;
  using AddEvaluateConst<FunctionType>::Evaluate;
  using AddEvaluate<FunctionType>::Evaluate;
};

} // namespace optimization
} // namespace mlpack

#endif
