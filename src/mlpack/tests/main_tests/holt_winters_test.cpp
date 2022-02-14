/**
 * @file tests/main_tests/holt_winters_test.cpp
 * @author Suvarsha Chennareddy
 *
 * Test mlpackMain() of holt_winters_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/time_series_models/holt_winters_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(HoltWintersTestFixture)

/**
 * Make sure that assertion fails when seasonal period and dataset's size mismatch.
 */
    TEST_CASE_METHOD(HoltWintersTestFixture, "DimensionsTest",
        "[HoltWintersMainTest][BindingTests]")
{
    arma::mat dataset(9, 3, arma::fill::randu);
    SetInputParam("input", std::move(dataset));
    SetInputParam("period", 5);
    SetInputParam("numberOfForecasts", 1);
    //Throws an error beacuse the dataset must have atleast 2*period elements
    REQUIRE_THROWS(RUN_BINDING());
}

/**
 * Holt Winters  model requires atleast one value for making predictions. Thus,
 * making sure an error is thrown when dataset has zero elements.
 */
TEST_CASE_METHOD(HoltWintersTestFixture, "EmptyDataTest",
    "[HoltWintersMainTest][BindingTests]")
{
    arma::mat dataset(0, 3, arma::fill::randu);
    SetInputParam("input", std::move(dataset));
    SetInputParam("period", 1);
    SetInputParam("numberOfForecasts", 1);
    REQUIRE_THROWS(RUN_BINDING());
}
