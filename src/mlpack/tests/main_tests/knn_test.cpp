/**
 * @file knn_test.cpp
 * @author Atharva Khandait
 *
 * Test mlpackMain() of knn_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "K-NearestNeighborsSearch";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/neighbor_search/knn_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KNNTestFixture
{
 public:
  KNNTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~KNNTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(KNNMainTest, KNNTestFixture);

/**
 * Make sure leaf size is always a non-negative number.
 */
BOOST_AUTO_TEST_CASE(KNNLeafSizeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 1000); // 1000 points in 3 dimensions.
  
  //Random input, negative leaf size.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();