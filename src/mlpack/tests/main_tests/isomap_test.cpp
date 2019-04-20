/**
 * @file isomap_test.cpp
 * @author Rishabh Ranjan
 *
 * Test for Isomap Main.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "Isomap";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/isomap/isomap_main.cpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;

struct IsomapFixture
{
 public:
  IsomapFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~IsomapFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(IsomapMainTest, IsomapFixture);

/**
 * Check that no invalid numbers of neighbours can be specified.
 */
BOOST_AUTO_TEST_CASE(IsomapInvalidNeighbourhoodSizeTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("neighbors", (int) 8); // more than number of data points

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that if number of neighbours specified produces a disconnected graph,
 * output is not calculated.
 */
BOOST_AUTO_TEST_CASE(IsomapNeighbourhoodGraphDisconnectedTest)
{
  // Example to produce disconnected neighbourhood graph.
  arma::mat x("0 1 8 9;"
              "0 1 8 9");

  SetInputParam("input", std::move(x));
  SetInputParam("neighbors", (int) 1); // too low

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
