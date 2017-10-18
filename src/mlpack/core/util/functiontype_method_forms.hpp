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
 * Definitions of method forms for the DecomposableFunctionType API.
 */
template <typename Class, typename...Ts>
using NumFunctionsForm = size_t(Class::*)(Ts...) const;

HAS_METHOD_FORM(NumFunctions, HasNumFunctions);

template <typename Class, typename...Ts>
using DecomposableEvaluateForm = double(Class::*)(const arma::mat&, const size_t,
    Ts...) const;

HAS_METHOD_FORM(Evaluate, HasDecomposableEvaluate);

template <typename Class, typename...Ts>
using DecomposableGradientForm = void(Class::*)(const arma::mat&, const size_t,
    arma::mat&, Ts...) const;

HAS_METHOD_FORM(Gradient, HasDecomposableGradient);

} // namespace static_checks
} // namespace mlpack

#endif
