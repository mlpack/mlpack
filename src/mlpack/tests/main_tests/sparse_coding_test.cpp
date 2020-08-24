/**
 * @file tests/main_tests/sparse_coding_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of sparse_coding_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "SparseCoding";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/sparse_coding/sparse_coding_main.cpp>
#include "test_helper.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

struct SparseCodingTestFixture
{
 public:
  SparseCodingTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~SparseCodingTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

/**
 * Helper function to load datasets.
 */
void LoadData(arma::mat& inputData, arma::mat& testData)
{
  // Load train dataset.
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load train dataset iris_train.csv!");

  // Load test dataset.
  if (!data::Load("iris_test.csv", testData))
    FAIL("Cannot load test dataset iris_test.csv!");
}

/**
 * Make sure that output points in dictionary equals number of
 * atoms passed and codes have desired dimension.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingOutputDimensionTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  REQUIRE(IO::GetParam<arma::mat>("dictionary").n_cols == 2);

  // Check that number of output dictionary rows equal number of input rows
  // which equal 4 for each data point.
  REQUIRE(IO::GetParam<arma::mat>("dictionary").n_rows == 4);

  // Check that number of output points are equal to number of test points.
  // Test file contains 63 data points.
  REQUIRE(IO::GetParam<arma::mat>("codes").n_cols == 63);

  // Check that number of output codes rows equal number of atoms.
  REQUIRE(IO::GetParam<arma::mat>("codes").n_rows == 2);
}

/**
 * Ensure that training data is normalized if normalize
 * parameter is set to true.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingNormalizationTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

  // Train for normalization set to false.

  // Reset passed parameters.
  bindings::tests::CleanMemory();
  IO::GetSingleton().Parameters()["normalize"].wasPassed = false;

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
  CheckMatrices(dictionary, IO::GetParam<arma::mat>("dictionary"));
  CheckMatrices(codes, IO::GetParam<arma::mat>("codes"));
}

/**
 * Ensure that l1, l2, max_iterations, objective_tolerance,
 * newton_tolerance value is always non-negative and number
 * of atoms is always positive.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingBoundsTest",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load train dataset iris_train.csv!");

  // Test for L1 value.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("lambda1", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for L2 value.

  // Input training data.
  bindings::tests::CleanMemory();
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("lambda2", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for max_iterations.

  // Input training data.
  bindings::tests::CleanMemory();
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for objective_tolerance.

  // Input training data.
  bindings::tests::CleanMemory();
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("objective_tolerance", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for newton_tolerance.

  // Input training data.
  bindings::tests::CleanMemory();
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("newton_tolerance", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for atoms.

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 0);

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure atoms are specified if training data is passed.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingReqAtomsTest",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("iris_train.csv", inputData))
    FAIL("Cannot load train dataset iris_train.csv!");

  // Input training data.
  SetInputParam("training", std::move(inputData));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure only one of input_model or initial_dictionary
 * is specified.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingModelVerTest",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);
  SparseCoding* c = new SparseCoding();

  // Input trained model and initial_dictionary.
  SetInputParam("input_model", c);
  SetInputParam("initial_dictionary", std::move(initialDictionary));

  Log::Fatal.ignoreInput = true;
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that specified number of atoms and initial_dictionary
 * atoms are equal.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingAtomsVerTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that input data and initial_dictionary
 * have same number of rows.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingRowsVerTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that training data and test data
 * have same dimensionality w.r.t rows.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingDataDimensionalityTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that saved model can be reused again.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingModelReuseTest",
                 "[SparseCodingMainTest][BindingTests]")
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
      std::move(IO::GetParam<arma::mat>("dictionary"));
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

  // Reset passed parameters.
  IO::GetSingleton().Parameters()["training"].wasPassed = false;

  // Test the correctness of trained model.

  // Input data.
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("input_model", IO::GetParam<SparseCoding*>("output_model"));
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output dictionary points are equals number of atoms.
  REQUIRE(IO::GetParam<arma::mat>("dictionary").n_cols == 2);

  // Check that number of output dictionary rows equal number of input rows
  // which equal 4 for each data point.
  REQUIRE(IO::GetParam<arma::mat>("dictionary").n_rows == 4);

  // Check that number of output points are equal to number of test points.
  // Test file contains 63 data points.
  REQUIRE(IO::GetParam<arma::mat>("codes").n_cols == 63);

  // Check that number of output codes rows equal number of atoms.
  REQUIRE(IO::GetParam<arma::mat>("codes").n_rows == 2);

  // Check that initial outputs and final outputs
  // using two models model are same.
  CheckMatrices(dictionary, IO::GetParam<arma::mat>("dictionary"));
  CheckMatrices(codes, IO::GetParam<arma::mat>("codes"));
}

/**
 * Ensure that for different value of max iterations
 * outputs are different.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingDiffMaxItrTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

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
  REQUIRE(arma::accu(dictionary ==
      IO::GetParam<arma::mat>("dictionary")) < dictionary.n_elem);

  REQUIRE(arma::accu(codes ==
      IO::GetParam<arma::mat>("codes")) < codes.n_elem);
}

/**
 * Ensure that for different value of objective_tolerance
 * outputs are different.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingDiffObjToleranceTest",
                 "[SparseCodingMainTest][BindingTests]")
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
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

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
  REQUIRE(arma::accu(dictionary ==
      IO::GetParam<arma::mat>("dictionary")) < dictionary.n_elem);
  REQUIRE(arma::accu(codes ==
      IO::GetParam<arma::mat>("codes")) < codes.n_elem);
}

/**
 * Ensure that for different value of newton_tolerance
 * outputs are different.
 */
TEST_CASE_METHOD(SparseCodingTestFixture,
                 "SparseCodingDiffNewtonToleranceTest",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for default newton_tolerance.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

  // Train for newton_tolerance equals to 10000.0.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("newton_tolerance", (double) 10000.0);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  REQUIRE(arma::accu(dictionary ==
      IO::GetParam<arma::mat>("dictionary")) < dictionary.n_elem);

  REQUIRE(arma::accu(codes ==
      IO::GetParam<arma::mat>("codes")) < codes.n_elem);
}

/**
 * Ensure that for different value of lambda1
 * outputs are different.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingDiffL1Test",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for default lambda1.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

  // Train for lambda1 equals to 10000.0.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("lambda1", (double) 10000.0);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  REQUIRE(arma::accu(dictionary ==
      IO::GetParam<arma::mat>("dictionary")) < dictionary.n_elem);

  REQUIRE(arma::accu(codes ==
      IO::GetParam<arma::mat>("codes")) < codes.n_elem);
}

/**
 * Ensure that for different value of lambda2
 * outputs are different.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingDiffL2Test",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for default lambda2.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

  // Train for lambda2 equals to 10000.0.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("lambda2", (double) 10000.0);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  REQUIRE(arma::accu(dictionary ==
      IO::GetParam<arma::mat>("dictionary")) < dictionary.n_elem);

  REQUIRE(arma::accu(codes ==
      IO::GetParam<arma::mat>("codes")) < codes.n_elem);
}

/**
 * Ensure that for different value of lambda1 & lambda2
 * outputs are different.
 */
TEST_CASE_METHOD(SparseCodingTestFixture, "SparseCodingDiffL1L2Test",
                 "[SparseCodingMainTest][BindingTests]")
{
  arma::mat inputData;
  arma::mat testData;
  LoadData(inputData, testData);

  mat initialDictionary = inputData.cols(0, 1);

  // Train for default lambda2 & lambda1 equal to 10000.0.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 2);
  SetInputParam("lambda1", (double) 10000.0);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary = IO::GetParam<arma::mat>("dictionary");
  arma::mat codes =
      std::move(IO::GetParam<arma::mat>("codes"));

  // Train for lambda1 EQUALS 0.0 & lambda2 equals to 10000.0.

  // Input data.
  bindings::tests::CleanMemory();
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 2);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("lambda1", (double) 0.0);
  SetInputParam("lambda2", (double) 10000.0);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  REQUIRE(arma::accu(dictionary ==
      IO::GetParam<arma::mat>("dictionary")) < dictionary.n_elem);

  REQUIRE(arma::accu(codes ==
      IO::GetParam<arma::mat>("codes")) < codes.n_elem);
}
