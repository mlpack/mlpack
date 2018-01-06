/**
 * @file linear_regression_test.cpp
 * @author Eugene Freyman
 *
 * Test mlpackMain() of linear_regression_main.cpp.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "LinearRegression";

#include "../test_tools.hpp"

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/linear_regression/linear_regression_main.cpp>

#include <boost/test/unit_test.hpp>

using namespace mlpack;

// Utility function to set a parameter and mark it as passed, using copy
// semantics.
template<typename T>
void SetInputParam(const std::string& name, const T& value)
{
  CLI::GetParam<T>(name) = value;
  CLI::SetPassed(name);
}

// Utility function to set a parameter and mark it as passed, using move
// semantics.
template<typename T>
void SetInputParam(const std::string& name, T&& value)
{
  CLI::GetParam<T>(name) = std::move(value);
  CLI::SetPassed(name);
}

struct LinearRegressionTestFixture
{
 public:
  LinearRegressionTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LinearRegressionTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LinearRegressionMainTest, LinearRegressionTestFixture);

BOOST_AUTO_TEST_CASE(LinearRegressionWrongResponseSizeTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);
  arma::rowvec y = arma::randu<arma::rowvec>(4);

  SetInputParam("training", std::move(x));
  SetInputParam("training_responses", std::move(y));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
