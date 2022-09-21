/**
 * @file tests/main_tests/radical_test.cpp
 * @author Manish Kumar
 *
 * Test RUN_BINDING() of radical_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/radical/radical_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(RadicalTestFixture);

/**
 * Check that output Y and W matrix have valid dimensions.
 */
TEST_CASE_METHOD(RadicalTestFixture, "RadicalOutputDimensionTest",
                "[RadicalMainTest][BindingTests]")
{
  arma::mat input = arma::randu<arma::mat>(5, 3);

  SetInputParam("input", std::move(input));

  RUN_BINDING();

  // Check dimension of Y matrix.
  REQUIRE(params.Get<arma::mat>("output_ic").n_rows == 5);
  REQUIRE(params.Get<arma::mat>("output_ic").n_cols == 3);

  // Check dimension of W matrix.
  REQUIRE(params.Get<arma::mat>("output_unmixing").n_rows == 5);
  REQUIRE(params.Get<arma::mat>("output_unmixing").n_cols == 5);
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

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  CleanMemory();
  ResetSettings();

  // Test for noise_std_dev.

  SetInputParam("input", input);
  SetInputParam("noise_std_dev", (double) -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  CleanMemory();
  ResetSettings();

  // Test for angles.

  SetInputParam("input", input);
  SetInputParam("angles", (int) 0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  CleanMemory();
  ResetSettings();

  // Test for sweeps.

  SetInputParam("input", input);
  SetInputParam("sweeps", (int) -2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
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

  FixedRandomSeed();
  RUN_BINDING();

  arma::mat Y = params.Get<arma::mat>("output_ic");

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(input));
  SetInputParam("noise_std_dev", (double) 0.01);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == params.Get<arma::mat>("output_ic")) < Y.n_elem);
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

  FixedRandomSeed();
  RUN_BINDING();

  arma::mat Y = params.Get<arma::mat>("output_ic");

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(input));
  SetInputParam("replicates", (int) 10);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == params.Get<arma::mat>("output_ic")) < Y.n_elem);
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

  FixedRandomSeed();
  RUN_BINDING();

  arma::mat Y = params.Get<arma::mat>("output_ic");

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(input));
  SetInputParam("angles", (int) 20);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == params.Get<arma::mat>("output_ic")) < Y.n_elem);
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

  FixedRandomSeed();
  RUN_BINDING();

  arma::mat Y = params.Get<arma::mat>("output_ic");

  CleanMemory();
  ResetSettings();
  ResetSettings();

  SetInputParam("input", std::move(input));
  SetInputParam("sweeps", (int) 2);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output and final output using two models are different.
  REQUIRE(arma::accu(Y == params.Get<arma::mat>("output_ic")) < Y.n_elem);
}
