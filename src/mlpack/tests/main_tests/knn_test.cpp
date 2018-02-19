/**
 * @file knn_test.cpp
 * @author Roberto Hueso Gomez
 *
 * Test mlpackMain() of nbc_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>

static const std::string testName = "KNN";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/neighbor_search/knn_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace arma;

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
 * Ensure resulting matrices have expected shape.
 */
BOOST_AUTO_TEST_CASE(KNNShapeTest)
{
  mat reference = randu<mat>(6, 100);
  mat query = randu<mat>(6, 10);
  int k = 5;

  SetInputParam("reference", std::move(reference));
  SetInputParam("query", std::move(query));
  SetInputParam("k", k);
  
  mlpackMain();

  const mat& distances = CLI::GetParam<mat>("distances");
  const Mat<size_t>& neighbors = CLI::GetParam<Mat<size_t>>("neighbors");
  
  BOOST_REQUIRE_EQUAL(neighbors.n_rows, 5);
  BOOST_REQUIRE_EQUAL(neighbors.n_cols, 10);
  BOOST_REQUIRE_EQUAL(distances.n_rows, 5);
  BOOST_REQUIRE_EQUAL(distances.n_cols, 10);
}

BOOST_AUTO_TEST_SUITE_END();
