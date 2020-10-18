/**
 * @file tests/main_tests/radical_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of radical_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "Radical";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/radical/radical_main.cpp>

#include "../catch.hpp"

using namespace mlpack;

struct RadicalTestFixture
{
 public:
  RadicalTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~RadicalTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Check that output Y and W matrix have valid dimensions.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalOutputDimensionTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input = arma::randu<arma::mat>(5, 3);

  SetInputParam("input", std::move(input));

  mlpackMain();

  // Check dimension of Y matrix.
  REQUIRE(IO::GetParam<arma::mat>("output_ic").n_rows == 5);
  REQUIRE(IO::GetParam<arma::mat>("output_ic").n_cols == 3);

  // Check dimension of W matrix.
  REQUIRE(IO::GetParam<arma::mat>("output_unmixing").n_rows == 5);
  REQUIRE(IO::GetParam<arma::mat>("output_unmixing").n_cols == 5);
}

/**
 * Ensure that replicates & angles are always positive while as noise_std_dev
 * & sweep is always non-negative.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalBoundsTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input = arma::randu<arma::mat>(5, 3);

  // Test for replicates.

  SetInputParam("input", input);
  SetInputParam("replicates", (int) 0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for noise_std_dev.

  SetInputParam("input", input);
  SetInputParam("noise_std_dev", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for angles.

  SetInputParam("input", input);
  SetInputParam("angles", (int) 0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for sweeps.

  SetInputParam("input", input);
  SetInputParam("sweeps", (int) -2);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check the learning process by using different values of noise_std_dev
 * parameter.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffNoiseStdDevTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = IO::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("noise_std_dev", (double) 0.01);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
}

/**
 * Check the learning process by using different values of replicates parameter.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffReplicatesTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = IO::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("replicates", (int) 10);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
}

/**
 * Check the learning process by using different values of angles parameter.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffAnglesTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = IO::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("angles", (int) 20);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
}

/**
 * Check the learning process by using different values of sweeps parameter.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalDiffSweepsTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = IO::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("sweeps", (int) 2);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == IO::GetParam<arma::mat>("output_ic")) < Y.n_elem);
}
