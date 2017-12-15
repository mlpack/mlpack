/**
 * @file linear_regression_test.cpp
 * @author Eugene Freyman
 *
 * Test mlpackMain() of linear_regression_main.cpp.
 */
#define BINDING_TYPE BINDING_TYPE_TEST
#define PROGRAM_NAME linearRegressionProgramName

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/linear_regression/linear_regression_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

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
    CLI::RestoreSettings(mlpack::bindings::tests::programName);
  }

  ~LinearRegressionTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LinearRegressionMainTest, LinearRegressionTestFixture);

BOOST_AUTO_TEST_CASE(LinearRegressionWringResponseSizeTest)
{
  std::cout << "1\n";
  SetInputParam("lambda", 1.0);
  std::cout << "2\n";
}

BOOST_AUTO_TEST_SUITE_END();
