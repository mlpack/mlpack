/**
 * @file local_coordinate_test.cpp
 * @author Neel Kelkar
 *
 * Test mlpackMain() of local_coordinate_coding_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "LocalCoordinate";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding_main.cpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LocalCoordinateTestFixture
{
 public:
  LocalCoordinateTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LocalCoordinateTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LocalCoordinateMainTest, LocalCoordinateTestFixture);

/**
 * Helper function to load datasets.
 */
void LoadData(arma::mat& inputData, arma::mat& testData)
{
  // Load train dataset.
  if (!data::Load("iris_train.csv", inputData))
    BOOST_FAIL("Cannot load train dataset iris_train.csv!");

  // Load test dataset.
  if (!data::Load("iris_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset iris_test.csv!");
}

/**
 * Make sure that output points in dictionary equals number of
 * atoms passed and codes have desired dimension.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateOutputDimensionTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Input data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output dictionary points are equals number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_cols, 2);

  // Check that number of output dictionary rows equal number of input rows
  // which equal 4 for each data point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_rows, 4);

  // Check that number of output points are equal to number of test points.
  // Test file contains 63 data points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_cols, 63);

  // Check that number of output codes rows equal number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_rows, 2);
}


/**
 * Ensure that training data is normalized if normalize
 * parameter is set to true.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateNormalizationTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for normalization set to true.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = CLI::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(CLI::GetParam<arma::mat>("codes"));

  // Train for normalization set to false.

  // Reset passed parameters.
  bindings::tests::CleanMemory();
  CLI::GetSingleton().Parameters()["normalize"].wasPassed = false;

  // Normalize train dataset.
  for (size_t i = 0; i < inputData.n_cols; ++i)
    inputData.col(i) /= norm(inputData.col(i), 2);

  // Normalize test dataset.
  for (size_t i = 0; i < testData.n_cols; ++i)
    testData.col(i) /= norm(testData.col(i), 2);

  // Input data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are same.
  CheckMatrices(dictionary, CLI::GetParam<arma::mat>("dictionary"));
  CheckMatrices(codes, CLI::GetParam<arma::mat>("codes"));
}

/**
 * Ensure that l, max_iterations, objective_tolerance,
 * newton_tolerance value is always non-negative and number
 * of atoms is always positive.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateBoundsTest)
{
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    BOOST_FAIL("Cannot load train dataset iris_train.csv!");

  // Test for Lambda value.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("lambda", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;


  // Test for max_iterations.

  // Input training data.
  bindings::tests::CleanMemory();
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for objective_tolerance.

  // Input training data.
  bindings::tests::CleanMemory();
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("objective_tolerance", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

 
  // Test for atoms.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure atoms are specified if training data is passed.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateReqAtomsTest)
{
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    BOOST_FAIL("Cannot load train dataset iris_train.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure only one of input_model or initial_dictionary
 * is specified.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateModelVerTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);
  LocalCoordinate* c = new LocalCoordinate();

  // Input trained model and initial_dictionary.
  SetInputParam("input_model", c);
  SetInputParam("initial_dictionary", std::move(initialDictionary));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that specified number of atoms and initial_dictionary
 * atoms are equal.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateAtomsVerTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1); // 2 points.

  // Input data and initial_dictionary.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 40); // Invalid.
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("max_iterations", (int) 100);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that input data and initial_dictionary
 * have same number of rows.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateRowsVerTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Trim inputData.
  inputData.shed_rows(1, 2);

  // Input data and initial_dictionary.
  SetInputParam("training", std::move(inputData)); // Invalid Data.
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that training data and test data
 * have same dimensionality w.r.t rows.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateDataDimensionalityTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Trim testData.
  testData.shed_rows(1, 2);

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("test", std::move(testData));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that saved model can be reused again.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateModelReuseTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary =
      std::move(CLI::GetParam<arma::mat>("dictionary"));
  arma::mat codes =
      std::move(CLI::GetParam<arma::mat>("codes"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;

  // Test the correctness of trained model.

  // Input data.
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("input_model", CLI::GetParam<LocalCoordinate*>("output_model"));
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output dictionary points are equals number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_cols, 2);

  // Check that number of output dictionary rows equal number of input rows
  // which equal 4 for each data point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_rows, 4);

  // Check that number of output points are equal to number of test points.
  // Test file contains 63 data points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_cols, 63);

  // Check that number of output codes rows equal number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_rows, 2);

  // Check that initial outputs and final outputs
  // using two models model are same.
  CheckMatrices(dictionary, CLI::GetParam<arma::mat>("dictionary"));
  CheckMatrices(codes, CLI::GetParam<arma::mat>("codes"));
}

/**
 * Ensure that for different value of max iterations
 * outputs are different.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateDiffMaxItrTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for max_iterations equals to 2.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("max_iterations", (int) 2);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = CLI::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(CLI::GetParam<arma::mat>("codes"));

  // Train for max_iterations equals to 100.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  BOOST_REQUIRE_LT(arma::accu(dictionary ==
      CLI::GetParam<arma::mat>("dictionary")), dictionary.n_elem);

  BOOST_REQUIRE_LT(arma::accu(codes ==
      CLI::GetParam<arma::mat>("codes")), codes.n_elem);
}

/**
 * Ensure that for different value of objective_tolerance
 * outputs are different.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateDiffObjToleranceTest)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for default objective_tolerance.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = CLI::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(CLI::GetParam<arma::mat>("codes"));

  // Train for objective_tolerance equals to 10000.0.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("objective_tolerance", (double) 10000.0);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  BOOST_REQUIRE_LT(arma::accu(dictionary ==
      CLI::GetParam<arma::mat>("dictionary")), dictionary.n_elem);

  BOOST_REQUIRE_LT(arma::accu(codes ==
      CLI::GetParam<arma::mat>("codes")), codes.n_elem);
}

/**
 * Ensure that for different value of lambda
 * outputs are different.
 */
BOOST_AUTO_TEST_CASE(LocalCoordinateDiffL2Test)
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for default lambda.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = CLI::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(CLI::GetParam<arma::mat>("codes"));

  // Train for lambda equals to 10000.0.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("lambda", (double) 10000.0);
  SetInputParam("test", std::move(testData));


  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  BOOST_REQUIRE_LT(arma::accu(dictionary ==
      CLI::GetParam<arma::mat>("dictionary")), dictionary.n_elem);

  BOOST_REQUIRE_LT(arma::accu(codes ==
      CLI::GetParam<arma::mat>("codes")), codes.n_elem);
}


BOOST_AUTO_TEST_SUITE_END();
