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
    arma::Row<double> dataset(9, arma::fill::randu);
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
    arma::Row<double> dataset(0,arma::fill::randu);
    SetInputParam("input", std::move(dataset));
    SetInputParam("period", 1);
    SetInputParam("numberOfForecasts", 1);
    REQUIRE_THROWS(RUN_BINDING());
}
