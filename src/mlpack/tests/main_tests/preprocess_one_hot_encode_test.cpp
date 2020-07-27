/**
 * @file preprocess_one_hot_encode_test.cpp
 * @author Jeffin Sam
 *
 * Test mlpackMain() of preprocess_one_hot_encoding_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessOneHotEncoding";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_one_hot_encoding_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct PreprocessOneHotEncodingTestFixture
{
 public:
  PreprocessOneHotEncodingTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessOneHotEncodingTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessOneHotEncodingMainTest,
                         PreprocessOneHotEncodingTestFixture);

BOOST_AUTO_TEST_CASE(PreprocessOneHotEncodingTest)
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  arma::mat matrix;
  matrix = "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;"
           "1 0 1 1 1 1 0 1;"
           "0 1 0 0 0 0 1 0;"
           "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {1, 3});
  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(matrix.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(matrix.n_rows, output.n_rows);  
  CheckMatrices(output, matrix);
}

BOOST_AUTO_TEST_CASE(EmptyMatrixTest)
{
  arma::mat dataset;
 
  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {1, 3});
  // error since dimesnions are bigger that matrix
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(EmptyIndicesTest)
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {});
  // error since dimesnions are bigger that matrix
  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(dataset.n_cols, output.n_cols);
  BOOST_REQUIRE_EQUAL(dataset.n_rows, output.n_rows);  
  CheckMatrices(output, dataset);
}
BOOST_AUTO_TEST_CASE(InvalidDimensionTest)
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {10000});
  // error since dimesnions are bigger that matrix
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_CASE(NegativeDimensionTest)
{
  arma::mat dataset;
  dataset = "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;"
            "-1 1 -1 -1 -1 -1 1 -1;"
            "1 1 -1 -1 -1 -1 1 1;";

  SetInputParam("input", dataset);
  SetInputParam<vector<int>>("dimensions", {-10000});
  // error since dimesnions are bigger that matrix
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
BOOST_AUTO_TEST_SUITE_END();
