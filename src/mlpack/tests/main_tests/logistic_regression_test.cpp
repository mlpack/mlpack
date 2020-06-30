/**
  * @file logistic_regression_test.cpp
  * @author B Kartheek Reddy
  *
  * Test mlpackMain() of logistic_regression_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "LogisticRegression";

#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>

using namespace mlpack;


struct LogisticRegressionTestFixture
{
 public:
  LogisticRegressionTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LogisticRegressionTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LogisticRegressionMainTest,
                         LogisticRegressionTestFixture);

/**
  * Ensuring that absence of training data is checked.
 **/
BOOST_AUTO_TEST_CASE(LRNoTrainingData)
{
  arma::Row<size_t> trainY;
  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr;

  SetInputParam("labels", std::move(trainY));

  // Training data is not provided. Should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that absence of responses is checked.
 */
BOOST_AUTO_TEST_CASE(LRNoResponses)
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  // Labels to the training data is not provided. It should throw
  // a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
BOOST_AUTO_TEST_CASE(LRPridictionSizeCheck)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;
  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr;
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", std::move(testX));

  // Training the model.
  mlpackMain();

  // Get the output predictions of the test data.
  const arma::Row<size_t> &testY =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Output predictions size must match the test data set size.
  BOOST_REQUIRE_EQUAL(testY.n_rows, 1);
  BOOST_REQUIRE_EQUAL(testY.n_cols, M);
}

/**
  * Ensuring that the response size is checked.
 **/
BOOST_AUTO_TEST_CASE(LRWrongResponseSizeTest)
{
  constexpr int D = 3;
  constexpr int N = 10;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY; // Response vector with wrong size.

  // 8 responses - incorrect size.
  trainY << 0 << 0 << 1 << 0 << 1 << 1 << 1 << 0 << arma::endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Labels with incorrect size. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking two options of specifying responses (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
BOOST_AUTO_TEST_CASE(LRResponsesRepresentationTest)
{
  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0, 1, 1}});
  arma::mat testX({{4.0, 5.0}, {1.0, 6.0}});

  SetInputParam("training", std::move(trainX1));
  SetInputParam("test", testX);

  // The first solution.
  mlpackMain();

  // Get the output.
  const arma::Row<size_t> testY1 =
      std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  // Now train by providing labels as extra parameter.
  arma::mat trainX2({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
  arma::Row<size_t> trainY2({0, 1, 1});

  SetInputParam("training", std::move(trainX2));
  SetInputParam("labels", std::move(trainY2));
  SetInputParam("test", std::move(testX));

  // The second solution.
  mlpackMain();

  // get the output
  const arma::Row<size_t> &testY2 =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Both solutions should be equal.
  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(),
                                  testY2.begin(), testY2.end());
}

/**
 * Check that model can saved / loaded and used. Ensuring that results are the
 * same.
 */
BOOST_AUTO_TEST_CASE(LRModelReload)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr;

  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", testX);

  // First solution
  mlpackMain();

  // Get the output model obtained from training.
  LogisticRegression<>* model =
      CLI::GetParam<LogisticRegression<>*>("output_model");
  // Get the output.
  const arma::Row<size_t> testY1 =
      std::move(CLI::GetParam<arma::Row<size_t>>("predictions"));

  // Reset the data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;
  CLI::GetSingleton().Parameters()["test"].wasPassed = false;

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  // Second solution.
  mlpackMain();

  // Get the output.
  const arma::Row<size_t> &testY2 =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Both solutions must be equal.
  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(),
                                  testY2.begin(), testY2.end());
}

/**
  * Checking for dimensionality of the test data set.
 **/
BOOST_AUTO_TEST_CASE(LRWrongDimOfTestData)
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr;

  // Test data with wrong dimensionality.
  arma::mat testX = arma::randu<arma::mat>(D-1, N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", std::move(testX));

  // Dimensionality of test data is wrong. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
BOOST_AUTO_TEST_CASE(LRWrongDimOfTestData2)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;
  // 10 responses
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training the model.
  mlpackMain();

  // Get the output model obtained from training.
  LogisticRegression<>* model =
      CLI::GetParam<LogisticRegression<>*>("output_model");

  // Reset the data passed.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["labels"].wasPassed = false;

  // Test data with Wrong dimensionality.
  arma::mat testX = arma::randu<arma::mat>(D - 1, M);
  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  // Test data dimensionality is wrong. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that training responses contain only two classes (0 or 1).
 **/
BOOST_AUTO_TEST_CASE(LRTrainWithMoreThanTwoClasses)
{
  constexpr int N = 8;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 8 responses containing more than two classes.
  trainY << 0 << 1 << 0 << 1 << 2 << 1 << 3 << 1 << arma::endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training data contains more than two classes. It should throw
  // a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that max iteration for optimizers is non negative.
 **/
BOOST_AUTO_TEST_CASE(LRNonNegativeMaxIterationTest)
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int(-1));

  // Maximum iterations is negative. It should a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that step size for optimizer is non negative.
 **/ 
BOOST_AUTO_TEST_CASE(LRNonNegativeStepSizeTest)
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << arma::endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("step_size", double(-0.01));

  // Step size for optimizer is negative. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that tolerance is non negative.
 **/
BOOST_AUTO_TEST_CASE(LRNonNegativeToleranceTest)
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 1 << 1 << 0 << 1 << 0 << 0 << 0 << 1 << 0 << 1 << arma::endr;

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("tolerance", double(-0.01));

  // Tolerance is negative. It should throw a runtime error.
  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring changing Maximum number of iterations changes the output model.
 **/
BOOST_AUTO_TEST_CASE(LRMaxIterationsChangeTest)
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 1 << 0 << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << arma::endr;

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("max_iterations", int(1));

  // First solution.
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(CLI::GetParam<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int(100));

  // Second solution.
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec &parameters2 =
      CLI::GetParam<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal
  // which ensures Max Iteration changes the output model.
  // arma::all function checks that each element of the vector is equal to zero.
  BOOST_REQUIRE_MESSAGE(!arma::all((parameters1-parameters2) == 0),
                        "Parameter(Max Iteration) has no effect on the output");
}

/**
  * Ensuring that lambda has some effects on the output.
 **/
BOOST_AUTO_TEST_CASE(LRLambdaChangeTest)
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 1 << 0 << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << arma::endr;

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("lambda", double(0));

  // First solution.
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(CLI::GetParam<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("lambda", double(1000));

  // Second solution.
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec &parameters2 =
      CLI::GetParam<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal
  // which ensures lambda changes the output model.
  // arma::all function checks that each element of the vector is equal to zero.
  BOOST_REQUIRE_MESSAGE(!arma::all((parameters1-parameters2) == 0),
                        "Parameter(lambda) has no effect on the output");
}

/**
  * Ensuring that Step size has some effects on the output.
 **/
BOOST_AUTO_TEST_CASE(LRStepSizeChangeTest)
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 1 << 0 << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << arma::endr;

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("step_size", double(0.02));

  // First solution.
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(CLI::GetParam<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("step_size", double(1.02));

  // Second solution.
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec &parameters2 =
      CLI::GetParam<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal
  // which ensures Step Size changes the output model.
  // arma::all function checks that each element of the vector is equal to zero.
  BOOST_REQUIRE_MESSAGE(!arma::all((parameters1-parameters2) == 0),
                        "Parameter(Step Size) has no effect on the output");
}

/**
  * Ensuring that lbfgs optimizer converges to a different result than sgd.
 **/
BOOST_AUTO_TEST_CASE(LROptimizerChangeTest)
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 1 << 0 << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << arma::endr;

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("optimizer", std::string("lbfgs"));
  SetInputParam("max_iterations", int(1000));

  // First solution.
  mlpackMain();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(CLI::GetParam<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("max_iterations", int(1000));

  // Second solution.
  mlpackMain();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec &parameters2 =
      CLI::GetParam<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal which
  // ensures that different optimizer converge to different results.
  // arma::all function checks that each element of the vector is equal to zero.
  BOOST_REQUIRE_MESSAGE(!arma::all((parameters1-parameters2) == 0),
                        "Parameter(Step Size) has no effect on the output");
}

/**
  * Ensuring decision_boundary parameter does something.
 **/
BOOST_AUTO_TEST_CASE(LRDecisionBoundaryTest)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY << 1 << 0 << 0 << 1 << 0 << 1 << 0 << 1 << 0 << 1 << arma::endr;

  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("decision_boundary", double(1));
  SetInputParam("test", testX);

  // First solution.
  mlpackMain();

  // Get the output after first training.
  const arma::Row<size_t> output1 =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Reset the settings.
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("decision_boundary", double(0));
  SetInputParam("test", testX);

  // Second solution.
  mlpackMain();

  // Get the output after second training.
  const arma::Row<size_t> &output2 =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Check that the output changed when the decision boundary moved.
  BOOST_REQUIRE_GT(arma::accu(output1 != output2), 0);
}

/**
  * Ensuring that the parameter 'output' and the parameter 'predictions' give
  * the same output
 **/
// The following test case is to check whether the old parameter 'output' and
// the new parameter 'predictions' give the same output
// This test case will be removed in mlpack 4
// when the deprecated parameter: 'output' is removed
BOOST_AUTO_TEST_CASE(LROPtionConsistencyTest){
  // Some data for training and testing
  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0, 1, 1}});
  arma::mat testX({{4.0, 5.0}, {1.0, 6.0}});

  SetInputParam("training", std::move(trainX1));
  SetInputParam("test", testX);

  // The solution.
  mlpackMain();

  // Get the output from 'predictions' parameter
  const arma::Row<size_t> testY1 =
      CLI::GetParam<arma::Row<size_t>>("predictions");

  // Get output from 'output' parameter
  const arma::Row<size_t> testY2 =
      std::move(CLI::GetParam<arma::Row<size_t>>("output"));

  // Both solutions must be equal.
  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(),
                                  testY2.begin(), testY2.end());
}

/**
  * Ensuring that the parameter 'output_probabilities' and the parameter
  * 'probabilities' give the same output
 **/
// The following test case is to check whether the old parameter
// 'output_probabilities' and the new parameter 'probabilities' give the same
// output. This test case will be removed in mlpack 4
// when the deprecated parameter: 'output_probabilities' is removed
BOOST_AUTO_TEST_CASE(LROPtionConsistencyTest2){
  // Some data for training and testing
  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0, 1, 1}});
  arma::mat testX({{4.0, 5.0}, {1.0, 6.0}});

  SetInputParam("training", std::move(trainX1));
  SetInputParam("test", testX);

  // The solution.
  mlpackMain();

  // Get the output from 'predictions' parameter
  const arma::mat testY1 =
      CLI::GetParam<arma::mat>("output_probabilities");

  // Get output from 'output' parameter
  const arma::mat testY2 =
      std::move(CLI::GetParam<arma::mat>("probabilities"));

  // Both solutions must be equal.
  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(),
                                  testY2.begin(), testY2.end());
}

BOOST_AUTO_TEST_SUITE_END();
