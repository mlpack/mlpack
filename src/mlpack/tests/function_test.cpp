/**
 * @file function_test.cpp
 * @author Ryan Curtin
 * @author Shikhar Bhardwaj
 *
 * Test the Function<> class to see that it properly adds functionality.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/function.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_function.hpp>
#include <mlpack/core/optimizers/sdp/sdp.hpp>
#include <mlpack/core/optimizers/sdp/lrsdp.hpp>
#include <mlpack/core/optimizers/aug_lagrangian/aug_lagrangian.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::optimization;
using namespace ens::traits; // For some SFINAE checks.
using namespace mlpack::regression;

/**
 * Utility class with no functions.
 */
class EmptyTestFunction { };

/**
 * Utility class with Evaluate() but no Evaluate().
 */
class EvaluateTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    return arma::accu(coordinates) + begin + batchSize;
  }
};

/**
 * Utility class with Gradient() but no Evaluate().
 */
class GradientTestFunction
{
 public:
  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  void Gradient(const arma::mat& coordinates,
                const size_t /* begin */,
                arma::mat& gradient,
                const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with Gradient() and Evaluate().
 */
class EvaluateGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  double Evaluate(const arma::mat& coordinates,
                  const size_t /* begin */,
                  const size_t /* batchSize */)
  {
    return arma::accu(coordinates);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  void Gradient(const arma::mat& coordinates,
                const size_t /* begin */,
                arma::mat& gradient,
                const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with EvaluateWithGradient().
 */
class EvaluateWithGradientTestFunction
{
 public:
  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }

  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t /* begin */,
                              arma::mat& gradient,
                              const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }
};

/**
 * Utility class with all three functions.
 */
class EvaluateAndWithGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates)
  {
    return arma::accu(coordinates);
  }

  double Evaluate(const arma::mat& coordinates,
                  const size_t begin,
                  const size_t batchSize)
  {
    return arma::accu(coordinates) + batchSize + begin;
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  void Gradient(const arma::mat& coordinates,
                const size_t /* begin */,
                arma::mat& gradient,
                const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }

  double EvaluateWithGradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }

  double EvaluateWithGradient(const arma::mat& coordinates,
                              const size_t /* begin */,
                              arma::mat& gradient,
                              const size_t /* batchSize */)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
    return arma::accu(coordinates);
  }
};

/**
 * Utility class with const Evaluate() and non-const Gradient().
 */
class EvaluateAndNonConstGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates) const
  {
    return arma::accu(coordinates);
  }

  void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

/**
 * Utility class with const Evaluate() and non-const Gradient().
 */
class EvaluateAndStaticGradientTestFunction
{
 public:
  double Evaluate(const arma::mat& coordinates) const
  {
    return arma::accu(coordinates);
  }

  static void Gradient(const arma::mat& coordinates, arma::mat& gradient)
  {
    gradient.ones(coordinates.n_rows, coordinates.n_cols);
  }
};

BOOST_AUTO_TEST_SUITE(FunctionTest);

/**
 * Make sure that an empty class doesn't have any methods added to it.
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientEmptyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<EmptyTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EmptyTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EmptyTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, false);
  BOOST_REQUIRE_EQUAL(hasGradient, false);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientEvaluateOnlyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<EvaluateTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EvaluateTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, false);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientGradientOnlyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<GradientTestFunction>,
                                       EvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<GradientTestFunction>,
                                       GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<GradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, false);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we add EvaluateWithGradient() when we have both Evaluate() and
 * Gradient().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientBothTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we add Evaluate() and Gradient() when we have only
 * EvaluateWithGradient().
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientEvaluateWithGradientTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateWithGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateWithGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateWithGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we add no methods when we already have all three.
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientAllThreeTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndWithGradientTestFunction>,
                           EvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndWithGradientTestFunction>,
                           GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndWithGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

BOOST_AUTO_TEST_CASE(LogisticRegressionEvaluateWithGradientTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<LogisticRegressionFunction<>>,
                           EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<LogisticRegressionFunction<>>,
                           GradientConstForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<LogisticRegressionFunction<>>,
                              EvaluateWithGradientConstForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

BOOST_AUTO_TEST_CASE(SDPTest)
{
  typedef AugLagrangianFunction<LRSDPFunction<SDP<arma::mat>>> FunctionType;

  const bool hasEvaluate =
      HasEvaluate<Function<FunctionType>, EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<FunctionType>, GradientConstForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<FunctionType>,
                              EvaluateWithGradientConstForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure that an empty class doesn't have any methods added to it.
 */
BOOST_AUTO_TEST_CASE(AddDecomposableEvaluateWithGradientEmptyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<EmptyTestFunction>,
                                       DecomposableEvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EmptyTestFunction>,
                                       DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EmptyTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, false);
  BOOST_REQUIRE_EQUAL(hasGradient, false);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we don't add any functions if we only have Evaluate().
 */
BOOST_AUTO_TEST_CASE(AddDecomposableEvaluateWithGradientEvaluateOnlyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<EvaluateTestFunction>,
                                       DecomposableEvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<EvaluateTestFunction>,
                                       DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, false);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we don't add any functions if we only have Gradient().
 */
BOOST_AUTO_TEST_CASE(AddDecomposableEvaluateWithGradientGradientOnlyTest)
{
  const bool hasEvaluate = HasEvaluate<Function<GradientTestFunction>,
                                       DecomposableEvaluateForm>::value;
  const bool hasGradient = HasGradient<Function<GradientTestFunction>,
                                       DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<GradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, false);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, false);
}

/**
 * Make sure we add EvaluateWithGradient() when we have both Evaluate() and
 * Gradient().
 */
BOOST_AUTO_TEST_CASE(AddDecomposableEvaluateWithGradientBothTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateGradientTestFunction>,
                           DecomposableEvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateGradientTestFunction>,
                           DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateGradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we add Evaluate() and Gradient() when we have only
 * EvaluateWithGradient().
 */
BOOST_AUTO_TEST_CASE(AddDecomposableEvaluateWGradientEvaluateWithGradientTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateWithGradientTestFunction>,
                           DecomposableEvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateWithGradientTestFunction>,
                           DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateWithGradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  Function<EvaluateWithGradientTestFunction> f;
  arma::mat coordinates(10, 10, arma::fill::ones);
  arma::mat gradient;
  f.Gradient(coordinates, 0, gradient, 5);

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we add no methods when we already have all three.
 */
BOOST_AUTO_TEST_CASE(AddDecomposableEvaluateWithGradientAllThreeTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndWithGradientTestFunction>,
                  DecomposableEvaluateForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndWithGradientTestFunction>,
                           DecomposableGradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndWithGradientTestFunction>,
                              DecomposableEvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we can properly create EvaluateWithGradient() even when one of the
 * functions is non-const.
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientMixedTypesTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndNonConstGradientTestFunction>,
                  EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndNonConstGradientTestFunction>,
                  GradientForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndNonConstGradientTestFunction>,
                              EvaluateWithGradientForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

/**
 * Make sure we can properly create EvaluateWithGradient() even when one of the
 * functions is static.
 */
BOOST_AUTO_TEST_CASE(AddEvaluateWithGradientMixedTypesStaticTest)
{
  const bool hasEvaluate =
      HasEvaluate<Function<EvaluateAndStaticGradientTestFunction>,
                  EvaluateConstForm>::value;
  const bool hasGradient =
      HasGradient<Function<EvaluateAndStaticGradientTestFunction>,
                  GradientStaticForm>::value;
  const bool hasEvaluateWithGradient =
      HasEvaluateWithGradient<Function<EvaluateAndStaticGradientTestFunction>,
                              EvaluateWithGradientConstForm>::value;

  BOOST_REQUIRE_EQUAL(hasEvaluate, true);
  BOOST_REQUIRE_EQUAL(hasGradient, true);
  BOOST_REQUIRE_EQUAL(hasEvaluateWithGradient, true);
}

class A
{
 public:
  size_t NumFunctions() const;
  size_t NumFeatures() const;
  double Evaluate(const arma::mat&, const size_t, const size_t) const;
  void Gradient(const arma::mat&, const size_t, arma::mat&, const size_t) const;
  void Gradient(const arma::mat&, const size_t, arma::sp_mat&, const size_t)
      const;
  void PartialGradient(const arma::mat&, const size_t, arma::sp_mat&) const;
};

class B
{
 public:
  size_t NumFunctions();
  size_t NumFeatures();
  double Evaluate(const arma::mat&, const size_t, const size_t);
  void Gradient(const arma::mat&, const size_t, arma::mat&, const size_t);
  void Gradient(const arma::mat&, const size_t, arma::sp_mat&, const size_t);
  void PartialGradient(const arma::mat&, const size_t, arma::sp_mat&);
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

/**
 * Test the correctness of the static check for LagrangianFunctionType API.
 */
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

/**
 * Test the correctness of the static check for SparseFunctionType API.
 */
BOOST_AUTO_TEST_CASE(SparseFunctionTypeCheckTest)
{
  static_assert(CheckSparseGradient<A>::value,
      "CheckSparseGradient static check failed.");
  static_assert(CheckSparseGradient<B>::value,
      "CheckSparseGradient static check failed.");
  static_assert(!CheckSparseGradient<C>::value,
      "CheckSparseGradient static check failed.");
  static_assert(!CheckSparseGradient<D>::value,
      "CheckSparseGradient static check failed.");
}

/**
 * Test the correctness of the static check for SparseFunctionType API.
 */
BOOST_AUTO_TEST_CASE(ResolvableFunctionTypeCheckTest)
{
  static_assert(CheckNumFeatures<A>::value,
      "CheckNumFeatures static check failed.");
  static_assert(CheckNumFeatures<B>::value,
      "CheckNumFeatures static check failed.");
  static_assert(!CheckNumFeatures<C>::value,
      "CheckNumFeatures static check failed.");
  static_assert(!CheckNumFeatures<D>::value,
      "CheckNumFeatures static check failed.");

  static_assert(CheckPartialGradient<A>::value,
      "CheckPartialGradient static check failed.");
  static_assert(CheckPartialGradient<B>::value,
      "CheckPartialGradient static check failed.");
  static_assert(!CheckPartialGradient<C>::value,
      "CheckPartialGradient static check failed.");
  static_assert(!CheckPartialGradient<D>::value,
      "CheckPartialGradient static check failed.");
}

BOOST_AUTO_TEST_SUITE_END();
