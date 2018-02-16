/**
 * @file static_checks.hpp
 * @author Shikhar Bhardwaj
 *
 * This file contains the definitions of the method forms required by the
 * FunctionType API used by the optimizers. These method forms can be used to
 * check the compliance of a user provided FunctionType with the required
 * interface from the optimizer at compile time.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_OPTIMIZERS_STATIC_CHECKS_HPP
#define MLPACK_CORE_OPTIMIZERS_STATIC_CHECKS_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {
namespace optimization {
namespace traits {

/**
 * Check if a suitable overload of Evaluate() is available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType>
struct CheckEvaluate
{
  const static bool value =
      HasEvaluate<FunctionType, EvaluateForm>::value ||
      HasEvaluate<FunctionType, EvaluateConstForm>::value ||
      HasEvaluate<FunctionType, EvaluateStaticForm>::value;
};

/**
 * Check if a suitable overload of Gradient() is available.
 *
 * This is required by the FunctionType API.
 */
template <typename FunctionType>
struct CheckGradient
{
  const static bool value =
      HasGradient<FunctionType, GradientForm>::value ||
      HasGradient<FunctionType, GradientConstForm>::value ||
      HasGradient<FunctionType, GradientStaticForm>::value;
};

/**
 * Check if a suitable overload of NumFunctions() is available.
 *
 * This is required by the DecomposableFunctionType API.
 */
template<typename FunctionType>
struct CheckNumFunctions
{
  const static bool value =
      HasNumFunctions<FunctionType, NumFunctionsForm>::value ||
      HasNumFunctions<FunctionType, NumFunctionsConstForm>::value ||
      HasNumFunctions<FunctionType, NumFunctionsStaticForm>::value;
};

/**
 * Check if a suitable overload of Shuffle() is available.
 *
 * This is required by the DecomposableFunctionType API.
 */
template<typename FunctionType>
struct CheckShuffle
{
  const static bool value =
      HasShuffle<FunctionType, ShuffleForm>::value ||
      HasShuffle<FunctionType, ShuffleConstForm>::value ||
      HasShuffle<FunctionType, ShuffleStaticForm>::value;
};

/**
 * Check if a suitable decomposable overload of Evaluate() is available.
 *
 * This is required by the DecomposableFunctionType API.
 */
template<typename FunctionType>
struct CheckDecomposableEvaluate
{
  const static bool value =
      HasEvaluate<FunctionType, DecomposableEvaluateForm>::value ||
      HasEvaluate<FunctionType, DecomposableEvaluateConstForm>::value ||
      HasEvaluate<FunctionType, DecomposableEvaluateStaticForm>::value;
};

/**
 * Check if a suitable decomposable overload of Gradient() is available.
 *
 * This is required by the DecomposableFunctionType API.
 */
template <typename FunctionType>
struct CheckDecomposableGradient
{
  const static bool value =
      HasGradient<FunctionType, DecomposableGradientForm>::value ||
      HasGradient<FunctionType, DecomposableGradientConstForm>::value ||
      HasGradient<FunctionType, DecomposableGradientStaticForm>::value;
};

/**
 * Check if a suitable overload of NumConstraints() is available.
 *
 * This is required by the ConstrainedFunctionType API.
 */
template<typename FunctionType>
struct CheckNumConstraints
{
  const static bool value =
      HasNumConstraints<FunctionType, NumConstraintsForm>::value ||
      HasNumConstraints<FunctionType, NumConstraintsConstForm>::value ||
      HasNumConstraints<FunctionType, NumConstraintsStaticForm>::value;
};

/**
 * Check if a suitable overload of EvaluateConstraint() is available.
 *
 * This is required by the ConstrainedFunctionType API.
 */
template<typename FunctionType>
struct CheckEvaluateConstraint
{
  const static bool value =
      HasEvaluateConstraint<FunctionType, EvaluateConstraintForm>::value ||
      HasEvaluateConstraint<FunctionType, EvaluateConstraintConstForm>::value ||
      HasEvaluateConstraint<FunctionType, EvaluateConstraintStaticForm>::value;
};

/**
 * Check if a suitable overload of GradientConstraint() is available.
 *
 * This is required by the ConstrainedFunctionType API.
 */
template <typename FunctionType>
struct CheckGradientConstraint
{
  const static bool value =
      HasGradientConstraint<FunctionType, GradientConstraintForm>::value ||
      HasGradientConstraint<FunctionType, GradientConstraintConstForm>::value ||
      HasGradientConstraint<FunctionType, GradientConstraintStaticForm>::value;
};

/**
 * Check if a suitable overload of Gradient() that supports sparse gradients is
 * available.
 *
 * This is required by the SparseFunctionType API.
 */
template <typename FunctionType>
struct CheckSparseGradient
{
  const static bool value =
      HasGradient<FunctionType, SparseGradientForm>::value ||
      HasGradient<FunctionType, SparseGradientConstForm>::value ||
      HasGradient<FunctionType, SparseGradientStaticForm>::value;
};

/**
 * Check if a suitable overload of NumFeatures() is available.
 *
 * This is required by the ResolvableFunctionType API.
 */
template<typename FunctionType>
struct CheckNumFeatures
{
  const static bool value =
      HasNumFeatures<FunctionType, NumFeaturesForm>::value ||
      HasNumFeatures<FunctionType, NumFeaturesConstForm>::value ||
      HasNumFeatures<FunctionType, NumFeaturesStaticForm>::value;
};

/**
 * Check if a suitable overload of PartialGradient() is available.
 *
 * This is required by the ResolvableFunctionType API.
 */
template <typename FunctionType>
struct CheckPartialGradient
{
  const static bool value =
      HasPartialGradient<FunctionType, PartialGradientForm>::value ||
      HasPartialGradient<FunctionType, PartialGradientConstForm>::value ||
      HasPartialGradient<FunctionType, PartialGradientStaticForm>::value;
};

/**
 * Check if a suitable overload of EvaluateWithGradient() is available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType>
struct CheckEvaluateWithGradient
{
  const static bool value =
      HasEvaluateWithGradient<FunctionType, EvaluateWithGradientForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          EvaluateWithGradientConstForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          EvaluateWithGradientStaticForm>::value;
};

/**
 * Check if a suitable decomposable overload of EvaluateWithGradient() is
 * available.
 *
 * This is required by the FunctionType API.
 */
template<typename FunctionType>
struct CheckDecomposableEvaluateWithGradient
{
  const static bool value =
      HasEvaluateWithGradient<FunctionType,
          DecomposableEvaluateWithGradientForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          DecomposableEvaluateWithGradientConstForm>::value ||
      HasEvaluateWithGradient<FunctionType,
          DecomposableEvaluateWithGradientStaticForm>::value;
};

/**
 * Perform checks for the regular FunctionType API.
 */
template<typename FunctionType>
inline void CheckFunctionTypeAPI()
{
  static_assert(CheckEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the FunctionType API; see the optimizer tutorial for details.");

  static_assert(CheckGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of Gradient(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the FunctionType API; see the optimizer tutorial for details.");

  static_assert(CheckEvaluateWithGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of "
      "EvaluateWithGradient().  Please check that the FunctionType fully "
      "satisfies the requirements of the FunctionType API; see the optimizer "
      "tutorial for more details.");
}

/**
 * Perform checks for the DecomposableFunctionType API.
 */
template<typename FunctionType>
inline void CheckDecomposableFunctionTypeAPI()
{
  static_assert(CheckDecomposableEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of a decomposable "
      "Evaluate() method.  Please check that the FunctionType fully satisfies"
      " the requirements of the DecomposableFunctionType API; see the optimizer"
      " tutorial for more details.");

  static_assert(CheckDecomposableGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of a decomposable "
      "Gradient() method.  Please check that the FunctionType fully satisfies"
      " the requirements of the DecomposableFunctionType API; see the optimizer"
      " tutorial for more details.");

  static_assert(CheckDecomposableEvaluateWithGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of a decomposable "
      "EvaluateWithGradient() method.  Please check that the FunctionType "
      "fully satisfies the requirements of the DecomposableFunctionType API; "
      "see the optimizer tutorial for more details.");

  static_assert(CheckNumFunctions<FunctionType>::value,
      "The FunctionType does not have a correct definition of NumFunctions(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the DecomposableFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckShuffle<FunctionType>::value,
      "The FunctionType does not have a correct definition of Shuffle(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the DecomposableFunctionType API; see the optimizer tutorial for more "
      "details.");
}

/**
 * Perform checks for the SparseFunctionType API.
 */
template<typename FunctionType>
inline void CheckSparseFunctionTypeAPI()
{
  static_assert(CheckNumFunctions<FunctionType>::value,
      "The FunctionType does not have a correct definition of NumFunctions(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the SparseFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckDecomposableEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the SparseFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckSparseGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of a sparse "
      "Gradient() method. Please check that the FunctionType fully satisfies "
      "the requirements of the SparseFunctionType API; see the optimizer "
      "tutorial for more details.");
}

/**
 * Perform checks for the NonDifferentiableFunctionType API.
 */
template<typename FunctionType>
inline void CheckNonDifferentiableFunctionTypeAPI()
{
  static_assert(CheckEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the NonDifferentiableFunctionType API; see the optimizer tutorial for "
      "more details.");
}

/**
 * Perform checks for the ResolvableFunctionType API.
 */
template<typename FunctionType>
inline void CheckResolvableFunctionTypeAPI()
{
  static_assert(CheckNumFeatures<FunctionType>::value,
      "The FunctionType does not have a correct definition of NumFeatures(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ResolvableFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ResolvableFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckPartialGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of a partial "
      "Gradient() function. Please check that the FunctionType fully satisfies "
      "the requirements of the ResolvableFunctionType API; see the optimizer "
      "tutorial for more details.");
}

/**
 * Perform checks for the ConstrainedFunctionType API.
 */
template<typename FunctionType>
inline void CheckConstrainedFunctionTypeAPI()
{
  static_assert(CheckEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckGradient<FunctionType>::value,
      "The FunctionType does not have a correct definition of Gradient(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckNumConstraints<FunctionType>::value,
      "The FunctionType does not have a correct definition of NumConstraints()."
      " Please check that the FunctionType fully satisfies the requirements of "
      "the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckEvaluateConstraint<FunctionType>::value,
      "The FunctionType does not have a correct definition of "
      "EvaluateConstraint(). Please check that the FunctionType fully satisfies"
      " the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");

  static_assert(CheckGradientConstraint<FunctionType>::value,
      "The FunctionType does not have a correct definition of "
      "GradientConstraint(). Please check that the FunctionType fully satisfies"
      " the ConstrainedFunctionType API; see the optimizer tutorial for more "
      "details.");
}

/**
 * Perform checks for the NonDifferentiableDecomposableFunctionType API.  (I
 * know, it is a long name...)
 */
template<typename FunctionType>
inline void CheckNonDifferentiableDecomposableFunctionTypeAPI()
{
  static_assert(CheckDecomposableEvaluate<FunctionType>::value,
      "The FunctionType does not have a correct definition of Evaluate(). "
      "Please check that the FunctionType fully satisfies the requirements of "
      "the NonDifferentiableDecomposableFunctionType API; see the optimizer "
      "tutorial for more details.");
}

} // namespace traits
} // namespace optimization
} // namespace mlpack

#endif
