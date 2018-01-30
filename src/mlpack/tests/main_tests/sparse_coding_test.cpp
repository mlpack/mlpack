/**
 * @file sparse_coding_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct SparseCodingTestFixture
{
 public:
  SparseCodingTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~SparseCodingTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(SparseCodingMainTest, SparseCodingTestFixture);

/**
 * Make sure that output points in dictionary equals number of
 * atoms passed and codes have desired dimension.
 */
BOOST_AUTO_TEST_CASE(SparseCodingOutputDimensionTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Generate test dataset.
  mat testData;
  testData = inputData.cols(450, 499);

  // Generate train dataset.
  inputData.shed_cols(450, 499);

  // Input data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 500);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output dictionary points are equals number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_cols, 30);

  // Check that number of output dictionary rows equal number of input rows
  // which equal 784 for each data point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_rows, 784);

  // Check that number of output points are equal to number of test points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_cols, 50);

  // Check that number of output codes rows equal number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_rows, 30);
}

/**
 * Ensure that training data is normalized if normalize
 * parameter is set to true.
 */
BOOST_AUTO_TEST_CASE(SparseCodingNormalizationTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Generate test dataset.
  mat testData;
  testData = inputData.cols(450, 499);

  // Generate train dataset.
  inputData.shed_cols(450, 499);

  // Generate initial dictionary.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 10);
  SetInputParam("normalize", (bool) true);

  mlpackMain();

  mat initialDictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));

  // Train for normalization set to true.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));
  arma::mat codes =
    std::move(CLI::GetParam<arma::mat>("codes"));

  // Train for normalization set to false.

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["normalize"].wasPassed = false;

  // Normalize train dataset.
  for (size_t i = 0; i < inputData.n_cols; ++i)
    inputData.col(i) /= norm(inputData.col(i), 2);

  // Normalize test dataset.
  for (size_t i = 0; i < testData.n_cols; ++i)
    testData.col(i) /= norm(testData.col(i), 2);

  // Input data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 30);
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
 * Ensure that l1, l2, max_iterations, objective_tolerance,
 * newton_tolerance value is always non-negative and number
 * of atoms is always positive.
 */
BOOST_AUTO_TEST_CASE(SparseCodingBoundsTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Test for L1 value.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("lambda1", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for L2 value.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("lambda2", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for max_iterations.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for objective_tolerance.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("objective_tolerance", (double) -1.0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // Test for newton_tolerance.

  // Input training data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 10);
  SetInputParam("newton_tolerance", (double) -1.0);

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
BOOST_AUTO_TEST_CASE(SparseCodingReqAtomsTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

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
BOOST_AUTO_TEST_CASE(SparseCodingModelVerTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Input data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 10);
  SetInputParam("normalize", (bool) true);

  mlpackMain();

  mat initialDictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));

  // Input trained model and initial_dictionary.
  SetInputParam("input_model",
                std::move(CLI::GetParam<SparseCoding>("output_model")));
  SetInputParam("initial_dictionary", std::move(initialDictionary));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that specified number of atoms and initial_dictionary
 * atoms are equal.
 */
BOOST_AUTO_TEST_CASE(SparseCodingAtomsVerTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 10);
  SetInputParam("normalize", (bool) true);

  mlpackMain();

  mat initialDictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));

  // Input data and initial_dictionary.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 40); // Invalid.
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that input data and initial_dictionary
 * have same number of rows.
 */
BOOST_AUTO_TEST_CASE(SparseCodingRowsVerTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);

  mlpackMain();

  mat initialDictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));

  // Trim inputData.
  inputData.shed_rows(100, 400);

  // Input data and initial_dictionary.
  SetInputParam("training", std::move(inputData)); // Invalid Data.
  SetInputParam("atoms", (int) 30);
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
BOOST_AUTO_TEST_CASE(SparseCodingDataDimensionalityTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Generate test dataset.
  mat testData;
  testData = inputData.cols(450, 499);

  // Trim testData.
  testData.shed_rows(100, 400);

  // Generate train dataset.
  inputData.shed_cols(450, 499);

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that saved model can be reused again.
 */
BOOST_AUTO_TEST_CASE(SparseCodingModelReuseTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Generate test dataset.
  mat testData;
  testData = inputData.cols(450, 499);

  // Generate train dataset.
  inputData.shed_cols(450, 499);

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
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
  SetInputParam("input_model",
                std::move(CLI::GetParam<SparseCoding>("output_model")));
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output dictionary points are equals number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_cols, 30);

  // Check that number of output dictionary rows equal number of input rows
  // which equal 784 for each data point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_rows, 784);

  // Check that number of output points are equal to number of test points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_cols, 50);

  // Check that number of output codes rows equal number of atoms.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_rows, 30);

  // Check that initial outputs and final outputs
  // using two models model are same.
  CheckMatrices(dictionary, CLI::GetParam<arma::mat>("dictionary"));
  CheckMatrices(codes, CLI::GetParam<arma::mat>("codes"));
}

/**
 * Ensure that for different value of max iterations
 * outputs are different.
 */
BOOST_AUTO_TEST_CASE(SparseCodingDiffMaxItrTest)
{
  mat inputData;
  inputData.load("mnist_first250_training_4s_and_9s.arm");

  // Shuffle input dataset.
  inputData = shuffle(inputData);

  // Generate test dataset.
  mat testData;
  testData = inputData.cols(450, 499);

  // Generate train dataset.
  inputData.shed_cols(450, 499);

  // Generate initial dictionary.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("max_iterations", (int) 1);
  SetInputParam("normalize", (bool) true);

  mlpackMain();

  mat initialDictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));

  // Train for max_iterations equals to 2.

  // Input data.
  SetInputParam("training", inputData);
  SetInputParam("atoms", (int) 30);
  SetInputParam("initial_dictionary", initialDictionary);
  SetInputParam("max_iterations", (int) 2);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", testData);

  mlpackMain();

  // Store outputs.
  arma::mat dictionary =
    std::move(CLI::GetParam<arma::mat>("dictionary"));
  arma::mat codes =
    std::move(CLI::GetParam<arma::mat>("codes"));

  // Train for max_iterations equals to 100.

  // Input data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("atoms", (int) 30);
  SetInputParam("initial_dictionary", std::move(initialDictionary));
  SetInputParam("max_iterations", (int) 100);
  SetInputParam("normalize", (bool) true);
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that initial outputs and final outputs
  // using two models model are different.
  for (size_t i = 0; i < dictionary.n_elem; ++i)
  {
    if(dictionary[i]!=0 && CLI::GetParam<arma::mat>("dictionary")[i]!=0)
    {
      BOOST_REQUIRE_NE(dictionary[i],
          CLI::GetParam<arma::mat>("dictionary")[i]);
    }
  }

  for (size_t i = 0; i < codes.n_elem; ++i)
  {
    if(codes[i]!=0 && CLI::GetParam<arma::mat>("codes")[i]!=0)
    {
      BOOST_REQUIRE_NE(codes[i],
          CLI::GetParam<arma::mat>("codes")[i]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
