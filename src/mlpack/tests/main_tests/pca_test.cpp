/**
 * @file pca_test.cpp
 * @author Ryan Curtin
 *
 * Test mlpackMain() of pca_main.cpp.
 */
#define BINDING_TYPE BINDING_TYPE_TEST
#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/pca/pca_main.cpp>

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

struct PCATestFixture
{
 public:
  PCATestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(mlpack::bindings::tests::programName);
  }

  ~PCATestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PCAMainTest, PCATestFixture);

/**
 * Make sure that if we ask for a dataset in 3 dimensions back, we get it.
 */
BOOST_AUTO_TEST_CASE(PCADimensionTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  // Random input, new dimensionality of 3.
  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 3);

  mlpackMain();

  // Now check that the output has 3 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 3);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
}

/**
 * Ensure that if we retain all variance, we get back a matrix with the same
 * dimensionality.
 */
BOOST_AUTO_TEST_CASE(PCAVarRetainTest)
{
  arma::mat x = arma::randu<arma::mat>(4, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("var_to_retain", (double) 1.0);
  SetInputParam("scale", true);
  SetInputParam("new_dimensionality", (int) 3); // Should be ignored.

  mlpackMain();

  // Check that the output has 5 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 4);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
}

/**
 * Ensure that if we retain no variance, we get back no dimensions.
 */
BOOST_AUTO_TEST_CASE(PCANoVarRetainTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("var_to_retain", (double) 0.01);
  SetInputParam("scale", true);
  SetInputParam("new_dimensionality", (int) 3); // Should be ignored.

  mlpackMain();

  // Check that the output has 1 dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 1);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
}

/**
 * Check that we can't specify an invalid new dimensionality.
 */
BOOST_AUTO_TEST_CASE(PCATooHighNewDimensionalityTest)
{
  arma::mat x = arma::randu<arma::mat>(5, 5);

  SetInputParam("input", std::move(x));
  SetInputParam("new_dimensionality", (int) 7); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
