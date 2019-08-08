/**
 * @file radical_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct RadicalTestFixture
{
 public:
  RadicalTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~RadicalTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(RadicalMainTest, RadicalTestFixture);

/**
 * Check that output Y and W matrix have valid dimensions.
 */
BOOST_AUTO_TEST_CASE(RadicalOutputDimensionTest)
{
  arma::mat input = arma::randu<arma::mat>(5, 3);

  SetInputParam("input", std::move(input));

  mlpackMain();

  // Check dimension of Y matrix.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output_ic").n_rows, 5);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output_ic").n_cols, 3);

  // Check dimension of W matrix.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output_unmixing").n_rows, 5);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output_unmixing").n_cols, 5);
}

/**
 * Ensure that replicates & angles are always positive while as noise_std_dev
 * & sweep is always non-negative.
 */
BOOST_AUTO_TEST_CASE(RadicalBoundsTest)
{
  arma::mat input = arma::randu<arma::mat>(5, 3);

  // Test for replicates.

  SetInputParam("input", input);
  SetInputParam("replicates", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for noise_std_dev.

  SetInputParam("input", input);
  SetInputParam("noise_std_dev", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for angles.

  SetInputParam("input", input);
  SetInputParam("angles", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  bindings::tests::CleanMemory();

  // Test for sweeps.

  SetInputParam("input", input);
  SetInputParam("sweeps", (int) -2);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check the learning process by using different values of noise_std_dev
 * parameter.
 */
BOOST_AUTO_TEST_CASE(RadicalDiffNoiseStdDevTest)
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = CLI::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("noise_std_dev", (double) 0.01);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  BOOST_REQUIRE_LT(arma::accu(Y ==
      CLI::GetParam<arma::mat>("output_ic")), Y.n_elem);
}

/**
 * Check the learning process by using different values of replicates parameter.
 */
BOOST_AUTO_TEST_CASE(RadicalDiffReplicatesTest)
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = CLI::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("replicates", (int) 10);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  BOOST_REQUIRE_LT(arma::accu(Y ==
      CLI::GetParam<arma::mat>("output_ic")), Y.n_elem);
}

/**
 * Check the learning process by using different values of angles parameter.
 */
BOOST_AUTO_TEST_CASE(RadicalDiffAnglesTest)
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = CLI::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("angles", (int) 20);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  BOOST_REQUIRE_LT(arma::accu(Y ==
      CLI::GetParam<arma::mat>("output_ic")), Y.n_elem);
}

/**
 * Check the learning process by using different values of sweeps parameter.
 */
BOOST_AUTO_TEST_CASE(RadicalDiffSweepsTest)
{
  arma::mat input("0.497369 0.891621 0.565789;"
                  "0.33821 0.494571 0.491079;"
                  "0.424898 0.297599 0.475061;"
                  "0.285009 0.152635 0.878107;"
                  "0.321474 0.997979 0.42137");

  SetInputParam("input", input);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::mat Y = CLI::GetParam<arma::mat>("output_ic");

  bindings::tests::CleanMemory();

  SetInputParam("input", std::move(input));
  SetInputParam("sweeps", (int) 2);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output and final output using two models are different.
  BOOST_REQUIRE_LT(arma::accu(Y ==
      CLI::GetParam<arma::mat>("output_ic")), Y.n_elem);
}

BOOST_AUTO_TEST_SUITE_END();
