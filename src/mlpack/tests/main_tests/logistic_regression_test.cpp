/**
  * @file logistic_regression_test.cpp
  * @author B Kartheek Reddy
  *
  * Test RUN_BINDING() of logistic_regression_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(LogisticRegressionTestFixture);

/**
  * Ensuring that absence of training data is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                 "LogisticRegressionLRNoTrainingData",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  arma::Row<size_t> trainY;
  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  SetInputParam("labels", std::move(trainY));

  // Training data is not provided. Should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that absence of responses is checked.
 */
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                 "LogisticRegressionLRNoResponses",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 1;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  SetInputParam("training", std::move(trainX));

  // Labels to the training data is not provided. It should throw
  // a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRPridictionSizeCheck",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;
  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", std::move(testX));

  // Training the model.
  RUN_BINDING();

  // Get the output predictions of the test data.
  const arma::Row<size_t>& testY =
      params.Get<arma::Row<size_t>>("predictions");

  // Output predictions size must match the test data set size.
  REQUIRE(testY.n_rows == 1);
  REQUIRE(testY.n_cols == M);
}

/**
  * Ensuring that the response size is checked.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                 "LogisticRegressionLRWrongResponseSizeTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int D = 3;
  constexpr int N = 10;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY; // Response vector with wrong size.

  // 8 responses - incorrect size.
  trainY = { 0, 0, 1, 0, 1, 1, 1, 0 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Labels with incorrect size. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Checking two options of specifying responses (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                 "LRResponsesRepresentationTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0, 1, 1}});
  arma::mat testX({{4.0, 5.0}, {1.0, 6.0}});

  SetInputParam("training", std::move(trainX1));
  SetInputParam("test", testX);

  // The first solution.
  RUN_BINDING();

  // Get the output.
  const arma::Row<size_t> testY1 =
      std::move(params.Get<arma::Row<size_t>>("predictions"));

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  // Now train by providing labels as extra parameter.
  arma::mat trainX2({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
  arma::Row<size_t> trainY2({0, 1, 1});

  SetInputParam("training", std::move(trainX2));
  SetInputParam("labels", std::move(trainY2));
  SetInputParam("test", std::move(testX));

  // The second solution.
  RUN_BINDING();

  // get the output
  const arma::Row<size_t>& testY2 =
      params.Get<arma::Row<size_t>>("predictions");

  // Both solutions should be equal.
  CheckMatrices(testY1, testY2);
}

/**
 * Check that model can saved / loaded and used. Ensuring that results are the
 * same.
 */
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                 "LogisticRegressionLRModelReload",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", testX);

  // First solution
  RUN_BINDING();

  // Get the output model obtained from training.
  LogisticRegression<>* model =
      params.Get<LogisticRegression<>*>("output_model");
  // Get the output.
  const arma::Row<size_t> testY1 =
      std::move(params.Get<arma::Row<size_t>>("predictions"));

  // Reset the data passed.
  ResetSettings();

  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  // Second solution.
  RUN_BINDING();

  // Get the output.
  const arma::Row<size_t>& testY2 =
      params.Get<arma::Row<size_t>>("predictions");

  // Both solutions must be equal.
  CheckMatrices(testY1, testY2);
}

/**
  * Checking for dimensionality of the test data set.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRWrongDimOfTestData",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  // Test data with wrong dimensionality.
  arma::mat testX = arma::randu<arma::mat>(D-1, N);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", std::move(testX));

  // Dimensionality of test data is wrong. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRWrongDimOfTestData2",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;
  // 10 responses
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training the model.
  RUN_BINDING();

  // Get the output model obtained from training.
  LogisticRegression<>* model =
      params.Get<LogisticRegression<>*>("output_model");

  // Reset the data passed.
  ResetSettings();

  // Test data with Wrong dimensionality.
  arma::mat testX = arma::randu<arma::mat>(D - 1, M);
  SetInputParam("input_model", model);
  SetInputParam("test", std::move(testX));

  // Test data dimensionality is wrong. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that training responses contain only two classes (0 or 1).
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                 "LRTrainWithMoreThanTwoClasses",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 8;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 8 responses containing more than two classes.
  trainY = { 0, 1, 0, 1, 2, 1, 3, 1 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // Training data contains more than two classes. It should throw
  // a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that max iteration for optimizers is non negative.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture,
                "LRNonNegativeMaxIterationTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int(-1));

  // Maximum iterations is negative. It should a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that step size for optimizer is non negative.
 **/ 
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRNonNegativeStepSizeTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("step_size", double(-0.01));

  // Step size for optimizer is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring that tolerance is non negative.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRNonNegativeToleranceTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 1, 0, 1, 0, 0, 0, 1, 0, 1 };

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("tolerance", double(-0.01));

  // Tolerance is negative. It should throw a runtime error.
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * Ensuring changing Maximum number of iterations changes the output model.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRMaxIterationsChangeTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 };

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("max_iterations", int(1));

  // First solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(params.Get<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int(100));

  // Second solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec& parameters2 =
      params.Get<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal
  // which ensures Max Iteration changes the output model.
  // arma::all function checks that each element of the vector is equal to zero.
  if (arma::all((parameters1-parameters2) == 0))
  {
    FAIL("parameters1 and parameters2 are equal. \
         Parameter(Max Iteration) has no effect on the output");
  }
}

/**
  * Ensuring that lambda has some effects on the output.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRLambdaChangeTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 };

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("lambda", double(0));

  // First solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(params.Get<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("lambda", double(1000));

  // Second solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec& parameters2 =
      params.Get<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal
  // which ensures lambda changes the output model.
  // arma::all function checks that each element of the vector is equal to zero.
  if (arma::all((parameters1-parameters2) == 0))
  {
    FAIL("parameters1 and parameters2 are equal. \
         Parameter(lambda) has no effect on the output");
  }
}

/**
  * Ensuring that Step size has some effects on the output.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRStepSizeChangeTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 };

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("step_size", double(0.02));

  // First solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 =
      std::move(params.Get<LogisticRegression<>*>("output_model")
                ->Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("step_size", double(1.02));

  // Second solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec& parameters2 =
      params.Get<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal
  // which ensures Step Size changes the output model.
  // arma::all function checks that each element of the vector is equal to zero.
  if (arma::all((parameters1-parameters2) == 0))
  {
    FAIL("parameters1 and parameters2 are equal. \
         Parameter(Step Size) has no effect on the output");
  }
}

/**
  * Ensuring that lbfgs optimizer converges to a different result than sgd.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LROptimizerChangeTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 };

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("optimizer", std::string("lbfgs"));
  SetInputParam("max_iterations", int(1000));

  // First solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after first training.
  const arma::rowvec parameters1 = std::move(
      params.Get<LogisticRegression<>*>("output_model")->Parameters());

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("optimizer", std::string("sgd"));
  SetInputParam("max_iterations", int(1000));

  // Second solution.
  RUN_BINDING();

  // Get the parameters of the output model obtained after second training.
  const arma::rowvec& parameters2 =
      params.Get<LogisticRegression<>*>("output_model")->Parameters();

  // Check that the parameters (parameters1 and parameters2) are not equal which
  // ensures that different optimizer converge to different results.
  // arma::all function checks that each element of the vector is equal to zero.
  if (arma::all((parameters1 - parameters2) == 0))
  {
    FAIL("parameters1 and parameters2 are equal. \
         Parameter(Step Size) has no effect on the output");
  }
}

/**
  * Ensuring decision_boundary parameter does something.
 **/
TEST_CASE_METHOD(LogisticRegressionTestFixture, "LRDecisionBoundaryTest",
                 "[LogisticRegressionMainTest][BindingTests]")
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  // 10 responses.
  trainY = { 1, 0, 0, 1, 0, 1, 0, 1, 0, 1 };

  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("decision_boundary", double(1));
  SetInputParam("test", testX);

  // First solution.
  RUN_BINDING();

  // Get the output after first training.
  const arma::Row<size_t> output1 =
      params.Get<arma::Row<size_t>>("predictions");

  // Reset the settings.
  CleanMemory();
  ResetSettings();

  SetInputParam("training", trainX);
  SetInputParam("labels", trainY);
  SetInputParam("decision_boundary", double(0));
  SetInputParam("test", testX);

  // Second solution.
  RUN_BINDING();

  // Get the output after second training.
  const arma::Row<size_t>& output2 =
      params.Get<arma::Row<size_t>>("predictions");

  // Check that the output changed when the decision boundary moved.
  REQUIRE(arma::accu(output1 != output2) > 0);
}
