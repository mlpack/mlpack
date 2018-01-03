
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "PreprocessSplit";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_split_main.cpp>

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

struct PreprocessSplitTestFixture
{
 public:
    PreprocessSplitTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessSplitTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessSplitMainTest, PreprocessSplitTestFixture);

/**
* Make sure that if we get a error if test ratio is invalid.
*/
    BOOST_AUTO_TEST_CASE(InvalidTestRatioSpecifiedErrorTest)
    {
      arma::mat x = arma::randu<arma::mat>(5, 5);
      SetInputParam("input", std::move(x));
      SetInputParam("test_ratio", 2.0);
      Log::Fatal.ignoreInput = true;
      BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
      Log::Fatal.ignoreInput = false;
    }


BOOST_AUTO_TEST_SUITE_END();
