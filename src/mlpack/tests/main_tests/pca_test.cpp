/**
 * @file tests/main_tests/pca_test.cpp
 * @author Ryan Curtin
 *
 * Test mlpackMain() of pca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "PrincipalComponentAnalysis";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/pca/pca_main.cpp>

#include "../catch.hpp"

using namespace mlpack;

struct PCATestFixture
{
 public:
  PCATestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~PCATestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Make sure that if we ask for a dataset in 3 dimensions back, we get it.
 */
TEST_CASE_METHOD(PCATestFixture, "PCADimensionTest",
                 "[PCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  // Random input, new dimensionality of 3.
  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);

  mlpackMain();

  // Now check that the output has 3 dimensions.
  REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 3);
  REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
}

/**
 * Ensure that if we retain all variance, we get back a matrix with the same
 * dimensionality.
 */
TEST_CASE_METHOD(PCATestFixture, "PCAVarRetainTest",
                 "[PCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(4, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("var_to_retain", (double) 1.0);
  SetInputParam("scale", true);
  SetInputParam("new_dimensionality", (int) 3); // Should be ignored.

  mlpackMain();

  // Check that the output has 5 dimensions.
  REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 4);
  REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
}

/**
 * Ensure that if we retain no variance, we get back no dimensions.
 */
TEST_CASE_METHOD(PCATestFixture, "PCANoVarRetainTest",
                 "[PCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("var_to_retain", (double) 0.01);
  SetInputParam("scale", true);
  SetInputParam("new_dimensionality", (int) 3); // Should be ignored.

  mlpackMain();

  // Check that the output has 1 dimensions.
  REQUIRE(IO::GetParam<arma::mat>("output").n_rows == 1);
  REQUIRE(IO::GetParam<arma::mat>("output").n_cols == 5);
}

/**
 * Check that we can't specify an invalid new dimensionality.
 */
TEST_CASE_METHOD(PCATestFixture, "PCATooHighNewDimensionalityTest",
                 "[PCAMainTest][BindingTests]")
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 7); // Invalid.

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
