/**
 * @file decision_tree_test.cpp
 * @author Manish Kumar
 *
 * Test mlpackMain() of decision_tree_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "DecisionTree";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/decision_tree/decision_tree_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

// Utility function to set a parameter and mark it as passed,
// using copy semantics for lvalues and move semantics for rvalues.
template<typename T>
void SetInputParam(const std::string& name, T&& value)
{
  CLI::GetParam<typename std::remove_reference<T>::type>(name)
       = std::forward<T>(value);
  CLI::SetPassed(name);
}

struct DecisionTreeTestFixture
{
 public:
  DecisionTreeTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~DecisionTreeTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(DecisionTreeMainTest,
                       DecisionTreeTestFixture);

/**
 * Check that number of output points and
 * number of input points are equal.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeOutputDimensionTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  // Input test data.
  SetInputParam("test", std::move(testData));

  mlpackMain();

  // Check that number of output points
  // are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
                      testSize);
}

/**
 * Make sure minimum leaf size is always a non-negative number.
 */
BOOST_AUTO_TEST_CASE(DecisionTreeMinimumLeafSizeTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  SetInputParam("minimum_leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(DecisionModelReuseTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Cannot load train dataset vc2.csv!");

  arma::Row<size_t> labels;
  if (!data::Load("vc2_labels.txt", labels))
    BOOST_FAIL("Cannot load labels for vc2_labels.txt");

  // Initialize an all-ones weight matrix.
  arma::mat weights(1, labels.n_cols, arma::fill::ones);

  // Input training data.
  SetInputParam("training", std::move(inputData));
  SetInputParam("labels", std::move(labels));
  SetInputParam("weights", std::move(weights));

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["weights"].wasPassed = false;

  arma::mat testData;
  if (!data::Load("vc2_test.csv", testData))
    BOOST_FAIL("Cannot load test dataset vc2.csv!");

  size_t testSize = testData.n_cols;

  // Input test data.
  SetInputParam("test", std::move(testData));
  SetInputParam("input_model",
                std::move(CLI::GetParam<DecisionTreeModel>("output_model")));

  mlpackMain();

  // Check that number of output points
  // are equal to number of input points.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Row<size_t>>("predictions").n_cols,
                      testSize);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("probabilities").n_cols,
                      testSize);
}

BOOST_AUTO_TEST_SUITE_END();
