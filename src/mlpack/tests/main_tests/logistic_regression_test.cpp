/**
  * @file logistic_regression_test.cpp
  * @author B Kartheek Reddy
  *
  * Test mlpackMain() of logistic_regression_main.cpp.
 **/

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
  	try { 
    	// Cache in the options for this program.
    	CLI::RestoreSettings(testName);
    } catch (std::invalid_argument e) {
    	Log::Fatal << "Invalid Test Name : " << e.what() << std::endl;
    }
  
  }

  ~LogisticRegressionTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }

};

BOOST_FIXTURE_TEST_SUITE(LogisticRegressionMainTest, LogisticRegressionTestFixture);

/** 
  * Ensuring that absence of training data is checked.
 **/
BOOST_AUTO_TEST_CASE(LRNoTrainingData) 
{
  arma::rowvec trainY;
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr; // 10 responses
  
  SetInputParam("training_responses", std::move(trainY));

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
  arma::rowvec trainY;
  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr; // 10 responses
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("training_responses", std::move(trainY));
  SetInputParam("test", std::move(testX));

  mlpackMain();

  const arma::rowvec testY = CLI::GetParam<arma::rowvec>("output_predictions");

  BOOST_REQUIRE_EQUAL(testY.n_rows, 1);
  BOOST_REQUIRE_EQUAL(testY.n_cols, M);
}

/**
  * Ensuring that the response size is checked
 **/
BOOST_AUTO_TEST_CASE(LRWrongResponseSizeTest) 
{
  constexpr int D = 3;
  constexpr int N = 10;
  
  arma::mat trainX = arma::randu<arma::mat>(D,N);
  arma::rowvec trainY; // response vector with wrong size

  trainY << 0 << 0 << 1 << 0 << 1 << 1 << 1 << 0 << arma::endr; // 8 responses - incorrect size

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(),std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking two options of specifying responses (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
BOOST_AUTO_TEST_CASE(LRResponsesRepresentationTest)
{
  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0,1,1}});
  arma::mat testX({{4.0,5.0},{1.0,6.0}});

  SetInputParam("training",std::move(trainX1));
  SetInputParam("test", testX);

  // The first solution.
  mlpackMain();

  const arma::rowvec testY1 = CLI::GetParam<arma::rowvec>("output");

  //reset the settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  arma::mat trainX2({{1.0, 2.0, 3.0},{1.0, 4.0, 9.0}});
  arma::rowvec trainY2({0,1,1});

  SetInputParam("training", std::move(trainX2));
  SetInputParam("labels", std::move(trainY2));
  SetInputParam("test", std::move(testX));

  // The second solution.
  mlpackMain();

  const arma::rowvec testY2 = CLI::GetParam<arma::rowvec>("output");

  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(), testY2.begin(), testY2.end());
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
  arma::rowvec trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr; // 10 responses
  
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", testX);

  //first solution
  mlpackMain();

  LogisticRegression<> model = CLI::GetParam<LogisticRegression<>>("output_model");
  const arma::rowvec testY1 = CLI::GetParam<arma::rowvec>("output");

  //reset the settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  //second solution
  mlpackMain();

  const arma::rowvec testY2 = CLI::GetParam<arma::rowvec>("output");

  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(), testY2.begin(), testY2.end());
}

/**
  * Checking for dimensionality of the test data set
 **/
BOOST_AUTO_TEST_CASE(LRWrongDimOfTestData) 
{
  constexpr int N = 10;
  constexpr int D = 4;

  arma::mat trainX = arma::randu<arma::mat>(D,N);
  arma::rowvec trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr; // 10 responses

  arma::mat testX = arma::randu<arma::mat>(D-1,N);  // test data with wrong dimensionality

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", std::move(testX));

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
  arma::rowvec trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr; // 10 responses

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // training the model
  mlpackMain();

  LogisticRegression<> model = CLI::GetParam<LogisticRegression<>>("output_model");

  //reset the settings
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);

  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // test data with Wrong dimensionality.
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that training responses contain only two classes (0 or 1)
 **/
BOOST_AUTO_TEST_CASE(LRTrainWithMoreThanTwoClasses) 
{
  constexpr int N = 8;
  constexpr int D = 2;

  arma::mat trainX = arma::randu<arma::mat>(D,N);
  arma::rowvec trainY;

  trainY << 0 << 1 << 0 << 1 << 2 << 1 << 3 << 1 << arma::endr; // 8 responses containing more than two classes

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(),std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * Ensuring that max iteration for optimizers is non negative 
 **/ 
BOOST_AUTO_TEST_CASE(LRMaxIterationNonNegativeTest) 
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D,N);
  arma::rowvec trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << arma::endr; // 10 responses

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int(-1));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

}

BOOST_AUTO_TEST_SUITE_END();


