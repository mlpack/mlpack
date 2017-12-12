#define BINDING_TYPE BINDING_TYPE_TEST
#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/emst/emst_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

namespace mlpack {
namespace bindings {
namespace tests {

extern std::string programName;

}
}
}

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

struct EMSTTestFixture
{
 public:
  EMSTTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(mlpack::bindings::tests::programName);
  }

  ~EMSTTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(EMSTMainTest, EMSTTestFixture);

/**
 * Make sure that Output has 3 Dimensions.
 */
BOOST_AUTO_TEST_CASE(EMSTOutputDimensionTest)
{
  arma::mat x = arma::randi<arma::mat>(6, 2, distr_param(0, 10));

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) 2);

  mlpackMain();

  // Now check that the output has 3 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 3);
}

/**
 * Check Naive algorithm Output has 3 Dimensions.
 */
BOOST_AUTO_TEST_CASE(EMSTNaiveOutputDimensionTest)
{
  arma::mat x = arma::randi<arma::mat>(6, 2, distr_param(0, 10));

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("naive", true);

  mlpackMain();

  // Now check that the output has 3 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 3);
}

/**
 * Ensure that we can't specify an invalid leaf size.
 */
BOOST_AUTO_TEST_CASE(EMSTInvalidLeafSizeTest)
{
  arma::mat x = arma::randi<arma::mat>(6, 2, distr_param(0, 10));

  // Input random data points.
  SetInputParam("input", std::move(x));
  SetInputParam("leaf_size", (int) -1); // Invalid leaf size.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error("leaf size must be greater than or equal to 1") );
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();