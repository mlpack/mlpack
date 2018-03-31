/**
 * @file preprocess_describe_test.cpp
 * @author Daivik Nema
 *
 * Test mlpackMain() of preprocess_describe_main.cpp.
 */

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "PreprocessDescribe";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/preprocess/preprocess_describe_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct PreprocessDescribeTestFixture
{
 public:
  PreprocessDescribeTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessDescribeTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessDescribeMainTest,
                         PreprocessDescribeTestFixture);

BOOST_AUTO_TEST_CASE(PreprocessDescribeDimensionInRangeCheck)
{
  // We will use the dataset trainSet.csv - which is already present in the data
  // directory. The dataset has 1000 rows and 24 columns. Invalid dimensions are
  // -1, -2 .. and so on
  // 1000, 1001 .. and so on
  int dimension = -1; // Invalid.
  arma::mat dataset;
  data::Load("trainSet.csv", dataset);

  SetInputParam("input", std::move(dataset));
  SetInputParam("dimension", dimension);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Now test with some value of dimension greater than max value.
  dimension = 1000; // Invalid.
  SetInputParam("dimension", dimension);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(PreprocessDescribePrecisionNonNegativeCheck)
{
  int precision = -1; // Invalid.
  arma::mat dataset;
  data::Load("trainSet.csv", dataset);

  SetInputParam("input", std::move(dataset));
  SetInputParam("precision", precision);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(PreprocessDescribeWidthNonNegativeTest)
{
  int width = -1; // Invalid.
  arma::mat dataset;
  data::Load("trainSet.csv", dataset);

  SetInputParam("input", std::move(dataset));
  SetInputParam("width", width);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(PreprocessDescribePopulationSampleTest)
{
  arma::mat dataset;
  data::Load("trainSet.csv", dataset);

  SetInputParam("input", std::move(dataset));
  mlpackMain();

  arma::mat sampleOutput = CLI::GetParam<arma::mat>("output");

  SetInputParam("population", true);
  mlpackMain();

  arma::mat populationOutput = CLI::GetParam<arma::mat>("output");

  BOOST_REQUIRE_EQUAL(sampleOutput.n_rows, populationOutput.n_rows);
  BOOST_REQUIRE_EQUAL(sampleOutput.n_cols, populationOutput.n_cols);
  BOOST_REQUIRE_EQUAL(sampleOutput.n_elem, populationOutput.n_elem);
  bool equal = true;
  for (size_t i=0; i < sampleOutput.n_elem; i++)
  {
    if (sampleOutput[i] != populationOutput[i])
    {
      equal = false;
      break;
    }
  }
  BOOST_REQUIRE_EQUAL(equal, false);
}

BOOST_AUTO_TEST_CASE(ProprocessDescribeRowMajorTest)
{
  arma::mat dataset;
  data::Load("trainSet.csv", dataset);

  SetInputParam("input", std::move(dataset));
  mlpackMain();

  arma::mat colMajorOutput = CLI::GetParam<arma::mat>("output");

  SetInputParam("row_major", true);
  mlpackMain();

  arma::mat rowMajorOutput = CLI::GetParam<arma::mat>("output");

  BOOST_REQUIRE_EQUAL(colMajorOutput.n_rows,
                      CLI::GetParam<arma::mat>("input").n_rows);
  BOOST_REQUIRE_EQUAL(rowMajorOutput.n_rows,
                      CLI::GetParam<arma::mat>("input").n_cols);
  BOOST_REQUIRE_EQUAL(colMajorOutput.n_cols, rowMajorOutput.n_cols);
}

BOOST_AUTO_TEST_SUITE_END();
