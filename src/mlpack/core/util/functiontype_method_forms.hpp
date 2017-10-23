/**
 * @file functiontype_form.hpp
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

#ifndef MLPACK_CORE_FUNCTIONTYPE_METHOD_FORMS
#define MLPACK_CORE_FUNCTIONTYPE_METHOD_FORMS

#include <mlpack/core.hpp>
#include <mlpack/core/util/sfinae_utility.hpp>

namespace mlpack {
namespace static_checks {
/*
 * Definitions of method forms and checks for the DecomposableFunctionType API.
 */
HAS_METHOD_FORM(NumFunctions, HasNumFunctions);

template <typename Class, typename...Ts>
using NumFunctionsFormConst = size_t(Class::*)() const;

template <typename Class, typename...Ts>
using NumFunctionsFormNonConst = size_t(Class::*)();

template<typename FunctionType>
struct CheckNumFunctions
{
  const static bool value =
    HasNumFunctions<FunctionType, NumFunctionsFormConst>::value ||
    HasNumFunctions<FunctionType, NumFunctionsFormNonConst>::value;
};

HAS_METHOD_FORM(Evaluate, HasDecomposableEvaluate);

template <typename Class, typename...Ts>
using DecomposableEvaluateFormConst = double(Class::*)(const arma::mat&, const
    size_t) const;

template <typename Class, typename...Ts>
using DecomposableEvaluateFormNonConst = double(Class::*)(const arma::mat&,
    const size_t);

template<typename FunctionType>
struct CheckDecomposableEvaluate
{
  const static bool value =
    HasDecomposableEvaluate<FunctionType, DecomposableEvaluateFormConst>::value
    || HasDecomposableEvaluate<FunctionType,
    DecomposableEvaluateFormNonConst>::value;
};

HAS_METHOD_FORM(Gradient, HasDecomposableGradient);

template <typename Class, typename...Ts>
using DecomposableGradientFormConst = void(Class::*)(const arma::mat&, const
    size_t, arma::mat&) const;

template <typename Class, typename...Ts>
using DecomposableGradientFormNonConst = void(Class::*)(const arma::mat&, const
    size_t, arma::mat&);

template <typename FunctionType>
struct CheckDecomposableGradient
{
  const static bool value =
    HasDecomposableGradient<FunctionType, DecomposableGradientFormConst>::value
    || HasDecomposableGradient<FunctionType,
    DecomposableGradientFormNonConst>::value;
};

/*
 * Definitions of method forms and checks for the LagrangianFunctionType API.
 */
HAS_METHOD_FORM(Evaluate, HasEvaluate);

template <typename Class, typename...Ts>
using EvaluateFormConst = double(Class::*)(const arma::mat&) const;

template <typename Class, typename...Ts>
using EvaluateFormNonConst = double(Class::*)(const arma::mat&);

template<typename FunctionType>
struct CheckEvaluate
{
  const static bool value = HasEvaluate<FunctionType, EvaluateFormConst>::value
    || HasEvaluate<FunctionType, EvaluateFormNonConst>::value;
};

HAS_METHOD_FORM(Gradient, HasGradient);

template <typename Class, typename...Ts>
using GradientFormConst = void(Class::*)(const arma::mat&, arma::mat&) const;

template <typename Class, typename...Ts>
using GradientFormNonConst = void(Class::*)(const arma::mat&, arma::mat&);

template <typename FunctionType>
struct CheckGradient
{
  const static bool value = HasGradient<FunctionType, GradientFormConst>::value
    || HasGradient<FunctionType, GradientFormNonConst>::value;
};

HAS_METHOD_FORM(NumConstraints, HasNumConstraints);

template <typename Class, typename...Ts>
using NumConstraintsFormConst = size_t(Class::*)() const;

template <typename Class, typename...Ts>
using NumConstraintsFormNonConst = size_t(Class::*)();

template<typename FunctionType>
struct CheckNumConstraints
{
  const static bool value =
    HasNumConstraints<FunctionType, NumConstraintsFormConst>::value ||
    HasNumConstraints<FunctionType, NumConstraintsFormNonConst>::value;
};

HAS_METHOD_FORM(EvaluateConstraint, HasEvaluateConstraint);

template <typename Class, typename...Ts>
using EvaluateConstraintFormConst = double(Class::*)(size_t, const arma::mat&)
  const;

template <typename Class, typename...Ts>
using EvaluateConstraintFormNonConst = double(Class::*)(size_t, const
    arma::mat&);

template<typename FunctionType>
struct CheckEvaluateConstraint
{
  const static bool value =
    HasEvaluateConstraint<FunctionType, EvaluateConstraintFormConst>::value ||
    HasEvaluateConstraint<FunctionType, EvaluateConstraintFormNonConst>::value;
};

HAS_METHOD_FORM(GradientConstraint, HasGradientConstraint);

template <typename Class, typename...Ts>
using GradientConstraintFormConst = void(Class::*)(size_t, const arma::mat&,
    arma::mat&) const;

template <typename Class, typename...Ts>
using GradientConstraintFormNonConst = void(Class::*)(size_t, const arma::mat&,
    arma::mat&);

template <typename FunctionType>
struct CheckGradientConstraint
{
  const static bool value =
    HasGradientConstraint<FunctionType, GradientConstraintFormConst>::value ||
    HasGradientConstraint<FunctionType, GradientConstraintFormNonConst>::value;
};

/*
 * Definitions of method forms and checks for the SparseFunctionType API.
 */
HAS_METHOD_FORM(Gradient, HasSparseGradient);

template <typename Class, typename...Ts>
using SparseGradientFormConst = void(Class::*)(const arma::mat&, size_t,
    arma::sp_mat&) const;

template <typename Class, typename...Ts>
using SparseGradientFormNonConst = void(Class::*)(const arma::mat&, size_t,
    arma::sp_mat&);

template <typename FunctionType>
struct CheckSparseGradient
{
  const static bool value =
    HasSparseGradient<FunctionType, SparseGradientFormConst>::value ||
    HasSparseGradient<FunctionType, SparseGradientFormNonConst>::value;
};

/**
 * Definitions of method forms and checks for the ResolvableFunctionType API.
 */
HAS_METHOD_FORM(NumFeatures, HasNumFeatures);

template <typename Class, typename...Ts>
using NumFeaturesFormConst = size_t(Class::*)() const;

template <typename Class, typename...Ts>
using NumFeaturesFormNonConst = size_t(Class::*)();

template<typename FunctionType>
struct CheckNumFeatures
{
  const static bool value =
    HasNumFeatures<FunctionType, NumFeaturesFormConst>::value ||
    HasNumFeatures<FunctionType, NumFeaturesFormNonConst>::value;
};

HAS_METHOD_FORM(PartialGradient, HasPartialGradient);

template <typename Class, typename...Ts>
using PartialGradientFormConst = void(Class::*)(const arma::mat&, size_t,
    arma::sp_mat&) const;

template <typename Class, typename...Ts>
using PartialGradientFormNonConst = void(Class::*)(const arma::mat&, size_t,
    arma::sp_mat&);

template <typename FunctionType>
struct CheckPartialGradient
{
  const static bool value =
    HasPartialGradient<FunctionType, PartialGradientFormConst>::value ||
    HasPartialGradient<FunctionType, PartialGradientFormNonConst>::value;
};

} // namespace static_checks
} // namespace mlpack

#endif
