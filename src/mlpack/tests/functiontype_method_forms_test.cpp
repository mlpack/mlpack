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
    double Evaluate(const arma::mat&) const;
    void Gradient(const arma::mat&, arma::mat&) const;
};

/**
 * Test the correctness of the method forms describing the
 * DecomposableFunctionType API.
 */
BOOST_AUTO_TEST_CASE(DecomposableFunctionTypeMethodFormsTest)
{
  static_assert(HasNumFunctions<A, NumFunctionsForm>::value,
      "NumFunctions static check failed.");
  static_assert(!HasNumFunctions<B, NumFunctionsForm>::value,
      "NumFunctions static check failed.");

  static_assert(HasDecomposableEvaluate<A, DecomposableEvaluateForm>::value,
      "DecomposableEvaluate static check failed");
  static_assert(!HasDecomposableEvaluate<B, DecomposableEvaluateForm>::value,
      "DecomposableEvaluate static check failed");

  static_assert(HasDecomposableGradient<A, DecomposableGradientForm>::value,
      "DecomposableGradient static check failed");
  static_assert(!HasDecomposableGradient<B, DecomposableGradientForm>::value,
      "DecomposableGradient static check failed");
}

BOOST_AUTO_TEST_SUITE_END();
