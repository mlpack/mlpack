/**
 * @file functiontype_method_forms.cpp
 * @author Shikhar Bhardwaj
 *
 * Test file for FunctionType method forms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/core/util/functiontype_method_forms.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::static_checks;

BOOST_AUTO_TEST_SUITE(FunctionTypeMethodFormsTest);

class A
{
 public:
  size_t NumFunctions() const;
  double Evaluate(const arma::mat&, const size_t) const;
  void Gradient(const arma::mat&, const size_t, arma::mat&) const;
};

class B
{
 public:
  size_t NumFunctions();
  double Evaluate(const arma::mat&, const size_t);
  void Gradient(const arma::mat&, const size_t, arma::mat&);
};

class C
{
  public:
    size_t NumConstraints() const;
    double Evaluate(const arma::mat&) const;
    void Gradient(const arma::mat&, arma::mat&) const;
    double EvaluateConstraint(const size_t, const arma::mat&) const;
    void GradientConstraint(const size_t, const arma::mat&, arma::mat&) const;
};

class D
{
 public:
  size_t NumConstraints();
  double Evaluate(const arma::mat&);
  void Gradient(const arma::mat&, arma::mat&);
  double EvaluateConstraint(const size_t, const arma::mat&);
  void GradientConstraint(const size_t, const arma::mat&, arma::mat&);
};


/**
 * Test the correctness of the static check for DecomposableFunctionType API.
 */

BOOST_AUTO_TEST_CASE(DecomposableFunctionTypeCheckTest)
{
  static_assert(CheckNumFunctions<A>::value,
      "CheckNumFunctions static check failed.");
  static_assert(CheckNumFunctions<B>::value,
      "CheckNumFunctions static check failed.");
  static_assert(!CheckNumFunctions<C>::value,
      "CheckNumFunctions static check failed.");
  static_assert(!CheckNumFunctions<D>::value,
      "CheckNumFunctions static check failed.");

  static_assert(CheckDecomposableEvaluate<A>::value,
      "CheckDecomposableEvaluate static check failed.");
  static_assert(CheckDecomposableEvaluate<B>::value,
      "CheckDecomposableEvaluate static check failed.");
  static_assert(!CheckDecomposableEvaluate<C>::value, 
      "CheckDecomposableEvaluate static check failed.");
  static_assert(!CheckDecomposableEvaluate<D>::value, 
      "CheckDecomposableEvaluate static check failed.");

  static_assert(CheckDecomposableGradient<A>::value,
      "CheckDecomposableGradient static check failed.");
  static_assert(CheckDecomposableGradient<B>::value,
      "CheckDecomposableGradient static check failed.");
  static_assert(!CheckDecomposableGradient<C>::value,
      "CheckDecomposableGradient static check failed.");
  static_assert(!CheckDecomposableGradient<D>::value, 
      "CheckDecomposableGradient static check failed.");
}

BOOST_AUTO_TEST_CASE(LagrangianFunctionTypeCheckTest)
{
  static_assert(!CheckEvaluate<A>::value, "CheckEvaluate static check failed.");
  static_assert(!CheckEvaluate<B>::value, "CheckEvaluate static check failed.");
  static_assert(CheckEvaluate<C>::value, "CheckEvaluate static check failed.");
  static_assert(CheckEvaluate<D>::value, "CheckEvaluate static check failed.");

  static_assert(!CheckGradient<A>::value, "CheckGradient static check failed.");
  static_assert(!CheckGradient<B>::value, "CheckGradient static check failed.");
  static_assert(CheckGradient<C>::value, "CheckGradient static check failed.");
  static_assert(CheckGradient<D>::value, "CheckGradient static check failed.");

  static_assert(!CheckNumConstraints<A>::value,
      "CheckNumConstraints static check failed.");
  static_assert(!CheckNumConstraints<B>::value,
      "CheckNumConstraints static check failed.");
  static_assert(CheckNumConstraints<C>::value,
      "CheckNumConstraints static check failed.");
  static_assert(CheckNumConstraints<D>::value,
      "CheckNumConstraints static check failed.");

  static_assert(!CheckEvaluateConstraint<A>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(!CheckEvaluateConstraint<B>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(CheckEvaluateConstraint<C>::value,
      "CheckEvaluateConstraint static check failed.");
  static_assert(CheckEvaluateConstraint<D>::value,
      "CheckEvaluateConstraint static check failed.");

  static_assert(!CheckGradientConstraint<A>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(!CheckGradientConstraint<B>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(CheckGradientConstraint<C>::value,
      "CheckGradientConstraint static check failed.");
  static_assert(CheckGradientConstraint<D>::value,
      "CheckGradientConstraint static check failed.");
}

BOOST_AUTO_TEST_SUITE_END();
