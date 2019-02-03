/**
 * @file local_coordinate_coding_test.cpp
 * @author Bhavya Bahl
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
static const std::string testName = "LocalCoordinateCoding";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/local_coordinate_coding/local_coordinate_coding_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LCCTestFixture
{
  public:
  LCCTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LCCTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LCCMainTest, LCCTestFixture);

/**
 * Ensure that the dimensions of encoded test points and output dictionary are correct.
 */
BOOST_AUTO_TEST_CASE(LCCDimensionsTest)
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma::mat t = x;

  SetInputParam("training", std::move(x));
  SetInputParam("test",std::move(t));
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  mlpackMain();

  // Check that the output has correct dimensions.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_rows, 10);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("codes").n_cols, 500);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_rows, 784);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("dictionary").n_cols, 10);
}


/**
 * Ensure that trained model can be reused.
 */
BOOST_AUTO_TEST_CASE(LCCOutputModelTest)
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma::mat t = x;

  SetInputParam("training", std::move(x));
  SetInputParam("test", t);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  mlpackMain();

  // Get the encoded output and dictionary after training.
  arma::mat initCodes = CLI::GetParam<arma::mat>("codes");
  arma::mat initDict = CLI::GetParam<arma::mat>("dictionary");
  LocalCoordinateCoding* outputModel = std::move(CLI::GetParam<LocalCoordinateCoding*>("output_model"));

  CLI::GetSingleton().Parameters()["training"].wasPassed = false;

  SetInputParam("input_model", std::move(outputModel));
  SetInputParam("test", std::move(t));

  mlpackMain();

  // Get the codes and the dictionary after reusing the trained model.
  arma::mat newCodes = CLI::GetParam<arma::mat>("codes");
  arma::mat newDict = CLI::GetParam<arma::mat>("dictionary");

  CheckMatrices(initCodes, newCodes);
  CheckMatrices(initDict, newDict);
}

/**
 * Ensure that the number of rows in initial dictionary is same as 
 * the dimension of the points.
 */
BOOST_AUTO_TEST_CASE(LCCInitDictTrainTest)
{
  arma::mat x = {{1,1,1,1},{2,2,2,2},{3,3,3,3},{4,4,4,4}};
  arma::mat initDict = {{1,1},{2,2},{3,3}};

  SetInputParam("training", std::move(x));
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("atoms", (int) 2);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that the number of columns in initial dictionary is same as 
 * the number of atoms.
 */
BOOST_AUTO_TEST_CASE(LCCInitDictAtomTest)
{
  arma::mat x = {{1,1,1,1},{2,2,2,2},{3,3,3,3},{4,4,4,4}};
  arma::mat initDict = {{1,1},{2,2},{3,3},{4,4}};

  SetInputParam("training", std::move(x));
  SetInputParam("initial_dictionary", std::move(initDict));
  SetInputParam("atoms", (int) 3);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that training data and test data points
 * have same dimensionality.
 */
BOOST_AUTO_TEST_CASE(LCCTrainAndTestDataDimTest)
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma::mat t = x;

  t.shed_rows(1,2);

  // Input data.
  SetInputParam("training", x);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);
  SetInputParam("test", std::move(t));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that only one out of training and input_model are specified.
 */
BOOST_AUTO_TEST_CASE(LCCTrainAndInputModelTest)
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  mlpackMain();

  LocalCoordinateCoding* outputModel = std::move(CLI::GetParam<LocalCoordinateCoding*>("output_model"));

  // No need to input training data again.
  SetInputParam("input_model", std::move(outputModel));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that dimensionality of the trained model matches
 * the dimensionality of the test points.
 */
BOOST_AUTO_TEST_CASE(LCCTrainedModelDimTest)
{
  arma::mat x;
  x.load("mnist_first250_training_4s_and_9s.arm");
  arma:: mat t = x;
  t.shed_rows(1,2);

  SetInputParam("training", x);
  SetInputParam("atoms", (int) 10);
  SetInputParam("max_iterations", (int) 2);

  mlpackMain();

  LocalCoordinateCoding* outputModel = std::move(CLI::GetParam<LocalCoordinateCoding*>("output_model"));

  SetInputParam("input_model", std::move(outputModel));
  SetInputParam("test", std::move(t));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();