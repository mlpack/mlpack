/**
 * @file tests/main_tests/persistence_test.cpp
 * @author Rishabh Garg
 *
 * Test mlpackMain() of persistence_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/time_series_models/persistence_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PersistenceTestFixture)

/**
 * Make sure that assertion fails when labels' and dataset's size mismatch.
 */
TEST_CASE_METHOD(PersistenceTestFixture, "DimensionsTest",
                 "[PersistenceMainTest][BindingTests]")
{
  arma::mat dataset(10, 3, arma::fill::randu);
  arma::rowvec labels(11, arma::fill::randu);

  SetInputParam("input", std::move(dataset));
  SetInputParam("labels", std::move(labels));

  REQUIRE_THROWS(RUN_BINDING());
}

/**
 * Persistence model requires atleast one row for making predictions. Thus,
 * making sure an error is thrown when dataset has zero rows.
 */
TEST_CASE_METHOD(PersistenceTestFixture, "EmptyDataTest",
                 "[PersistenceMainTest][BindingTests]")
{
  arma::mat dataset(0, 3, arma::fill::randu);
  SetInputParam("input", std::move(dataset));

  // Lag function inside `methods/time_series_models/util/lag.hpp` throws an
  // error because the number of periods to shift (1 in this case) is larger
  // than the number of rows in dataset.
  REQUIRE_THROWS(RUN_BINDING());
}