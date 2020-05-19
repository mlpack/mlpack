/**
  * @file nmf_test.cpp
  * @author Wenhao Huang
  *
  * Test mlpackMain() of nmf_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "NonNegativeMatrixFactorization";

#include <mlpack/core.hpp>
#include <mlpack/methods/nmf/nmf_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace arma;


struct NMFTestFixture
{
 public:
  NMFTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~NMFTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

static void ResetSettings()
{
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(NMFMainTest, NMFTestFixture);

/**
 * Ensure the resulting matrices W, H have expected shape.
 * Multdist update rule (Default Case).
 */
BOOST_AUTO_TEST_CASE(NMFMultdistShapeTest)
{
  mat v = randu<mat>(8, 10);
  int r = 5;

  SetInputParam("update_rules", std::string("multdist"));
  SetInputParam("input", std::move(v));
  SetInputParam("rank", r);

  // Perform NMF.
  mlpackMain();

  // Get resulting matrices.
  const mat& w = CLI::GetParam<mat>("w");
  const mat& h = CLI::GetParam<mat>("h");

  // Check the shapes of W and H.
  BOOST_REQUIRE_EQUAL(w.n_rows, 8);
  BOOST_REQUIRE_EQUAL(w.n_cols, 5);
  BOOST_REQUIRE_EQUAL(h.n_rows, 5);
  BOOST_REQUIRE_EQUAL(h.n_cols, 10);
}

/**
 * Ensure the resulting matrices W, H have expected shape.
 * Multdiv update rule.
 */
BOOST_AUTO_TEST_CASE(NMFMultdivShapeTest)
{
  mat v = randu<mat>(8, 10);
  int r = 5;

  SetInputParam("update_rules", std::string("multdiv"));
  SetInputParam("input", std::move(v));
  SetInputParam("rank", r);

  // Perform NMF.
  mlpackMain();

  // Get resulting matrices.
  const mat& w = CLI::GetParam<mat>("w");
  const mat& h = CLI::GetParam<mat>("h");

  // Check the shapes of W and H.
  BOOST_REQUIRE_EQUAL(w.n_rows, 8);
  BOOST_REQUIRE_EQUAL(w.n_cols, 5);
  BOOST_REQUIRE_EQUAL(h.n_rows, 5);
  BOOST_REQUIRE_EQUAL(h.n_cols, 10);
}

/**
 * Ensure the resulting matrices W, H have expected shape.
 * Als update rule.
 */
BOOST_AUTO_TEST_CASE(NMFAlsShapeTest)
{
  mat v = randu<mat>(8, 10);
  int r = 5;

  SetInputParam("update_rules", std::string("als"));
  SetInputParam("input", std::move(v));
  SetInputParam("rank", r);

  // Perform NMF.
  mlpackMain();

  // Get resulting matrices.
  const mat& w = CLI::GetParam<mat>("w");
  const mat& h = CLI::GetParam<mat>("h");

  // Check the shapes of W and H.
  BOOST_REQUIRE_EQUAL(w.n_rows, 8);
  BOOST_REQUIRE_EQUAL(w.n_cols, 5);
  BOOST_REQUIRE_EQUAL(h.n_rows, 5);
  BOOST_REQUIRE_EQUAL(h.n_cols, 10);
}

/**
 * Ensure the rank is positive.
 */
BOOST_AUTO_TEST_CASE(NMFRankBoundTest)
{
  mat v = randu<mat>(10, 10);
  int r;

  // Rank should not be negative.
  r = -1;
  SetInputParam("input", std::move(v));
  SetInputParam("rank", r);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Rank should not be 0.
  r = 0;
  SetInputParam("rank", r);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure the max_iterations is non-negative.
 */
BOOST_AUTO_TEST_CASE(NMFMaxIterartionBoundTest)
{
  mat v = randu<mat>(10, 10);
  int r = 5;

  // max_iterations should be non-negative.
  SetInputParam("max_iterations", int(-1));
  SetInputParam("input", std::move(v));
  SetInputParam("rank", r);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure the update rule is one of 
 * {"multdist", "multdiv", "als"}.
 */
BOOST_AUTO_TEST_CASE(NMFUpdateRuleTest)
{
  mat v = randu<mat>(10, 10);
  int r = 5;

  // Invalid update rule.
  SetInputParam("update_rules", std::string("invalid_rule"));
  SetInputParam("input", std::move(v));
  SetInputParam("rank", r);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure min_residue is used, by testing that 
 * min_resude makes a difference to the program.  
 */
BOOST_AUTO_TEST_CASE(NMFMinResidueTest)
{
  mat v = arma::randu(10, 10);
  mat initialW = arma::randu(10, 5);
  mat initialH = arma::randu(5, 10);
  int r = 5;

  // Set a larger min_residue.
  SetInputParam("min_residue", double(1));
  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_w", initialW);
  SetInputParam("initial_h", initialH);

  mlpackMain();

  const mat w1 = CLI::GetParam<mat>("w");
  const mat h1 = CLI::GetParam<mat>("h");

  ResetSettings();

  // Set a smaller min_residue.
  SetInputParam("min_residue", double(1e-3));
  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_w", initialW);
  SetInputParam("initial_h", initialH);

  mlpackMain();

  const mat w2 = CLI::GetParam<mat>("w");
  const mat h2 = CLI::GetParam<mat>("h");

  // The resulting matrices should be different.
  BOOST_REQUIRE_GT(arma::norm(w1 - w2), 1e-5);
  BOOST_REQUIRE_GT(arma::norm(h1 - h2), 1e-5);
}

/**
 * Ensure max_iterations is used, by testing that 
 * max_iterations makes a difference to the program.  
 */
BOOST_AUTO_TEST_CASE(NMFMaxIterationTest)
{
  mat v = arma::randu(10, 10);
  mat initialW = arma::randu(10, 5);
  mat initialH = arma::randu(5, 10);
  int r = 5;

  // Set a larger max_iterations.
  SetInputParam("max_iterations", int(100));
  // Remove the influence of min_residue.
  SetInputParam("min_residue", double(0));
  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_w", initialW);
  SetInputParam("initial_h", initialH);

  mlpackMain();

  const mat w1 = CLI::GetParam<mat>("w");
  const mat h1 = CLI::GetParam<mat>("h");

  ResetSettings();

  // Set a smaller max_iterations.
  SetInputParam("max_iterations", int(5));
  // Remove the influence of min_residue.
  SetInputParam("min_residue", double(0));
  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_w", initialW);
  SetInputParam("initial_h", initialH);

  mlpackMain();

  const mat w2 = CLI::GetParam<mat>("w");
  const mat h2 = CLI::GetParam<mat>("h");

  // The resulting matrices should be different.
  BOOST_REQUIRE_GT(arma::norm(w1 - w2), 1e-5);
  BOOST_REQUIRE_GT(arma::norm(h1 - h2), 1e-5);
}

/**
 * Test NMF with given initial_w and initial_h.
 */
BOOST_AUTO_TEST_CASE(NMFWHGivenInitTest)
{
  mat v = arma::randu(10, 10);
  mat initialW = arma::randu(10, 5);
  mat initialH = arma::randu(5, 10);
  int r = 5;

  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_w", initialW);
  SetInputParam("initial_h", initialH);

  mlpackMain();

  const mat w = CLI::GetParam<mat>("w");
  const mat h = CLI::GetParam<mat>("h");

  // Check the shapes of W and H.
  BOOST_REQUIRE_EQUAL(w.n_rows, 10);
  BOOST_REQUIRE_EQUAL(w.n_cols, 5);
  BOOST_REQUIRE_EQUAL(h.n_rows, 5);
  BOOST_REQUIRE_EQUAL(h.n_cols, 10);
}

/**
 * Test NMF with given initial_w.
 */
BOOST_AUTO_TEST_CASE(NMFWGivenInitTest)
{
  mat v = arma::randu(10, 10);
  mat initialW = arma::randu(10, 5);
  int r = 5;

  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_w", initialW);

  mlpackMain();

  const mat w = CLI::GetParam<mat>("w");
  const mat h = CLI::GetParam<mat>("h");

  // Check the shapes of W and H.
  BOOST_REQUIRE_EQUAL(w.n_rows, 10);
  BOOST_REQUIRE_EQUAL(w.n_cols, 5);
  BOOST_REQUIRE_EQUAL(h.n_rows, 5);
  BOOST_REQUIRE_EQUAL(h.n_cols, 10);
}

/**
 * Test NMF with given initial_h.
 */
BOOST_AUTO_TEST_CASE(NMFHGivenInitTest)
{
  mat v = arma::randu(10, 10);
  mat initialH = arma::randu(5, 10);
  int r = 5;

  SetInputParam("input", v);
  SetInputParam("rank", r);
  SetInputParam("initial_h", initialH);

  mlpackMain();

  const mat w = CLI::GetParam<mat>("w");
  const mat h = CLI::GetParam<mat>("h");

  // Check the shapes of W and H.
  BOOST_REQUIRE_EQUAL(w.n_rows, 10);
  BOOST_REQUIRE_EQUAL(w.n_cols, 5);
  BOOST_REQUIRE_EQUAL(h.n_rows, 5);
  BOOST_REQUIRE_EQUAL(h.n_cols, 10);
}


BOOST_AUTO_TEST_SUITE_END();

