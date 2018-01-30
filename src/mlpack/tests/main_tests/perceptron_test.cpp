/**
  * @file perceptron_test.cpp
  * @author B Kartheek Reddy
  *
  * Test mlpackMain() of perceptron_main.cpp.
 **/

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "PerceptronModel";


#include <mlpack/core.hpp>
#include <mlpack/methods/perceptron/perceptron_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct PerceptronTestFixture
{
	public:
	PerceptronTestFixture()
  	{
  		try { 
    		// Cache in the options for this program.
    		CLI::RestoreSettings(testName);
    	} catch (std::invalid_argument e) {
    		Log::Fatal << "Invalid Test Name" << e.what() << std::endl;
    	}
  
  	}

  	~PerceptronTestFixture()
  	{
    	// Clear the settings.
    	CLI::ClearSettings();
  	}

};

void resetSettings() {
	CLI::ClearSettings();
    CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(PerceptronMainTest, PerceptronTestFixture);

/**
  * Checking for dimensionality of the test data set
 **/
BOOST_AUTO_TEST_CASE(PerceptronWrongDimOfTestData) 
{
	constexpr int N = 10;
	constexpr int D = 4;
	constexpr int M = 20;

	arma::mat trainX = arma::randu<arma::mat>(D,N);
	arma::Row<size_t> trainY;

	trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr; // 10 responses

	arma::mat testX = arma::randu<arma::mat>(D-3,M);  // test data with wrong dimensionality

	SetInputParam("training", std::move(trainX));
	// SetInputParam("labels", std::move(trainY));
	SetInputParam("test", std::move(testX));

	Log::Fatal.ignoreInput = true;
	BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
	Log::Fatal.ignoreInput = false;

}

/**
  * Ensuring that re-training of an existing model with different of classes is checked
 **/
BOOST_AUTO_TEST_CASE(PerceptronReTrainWithWrongClasses)
{
	arma::mat trainX1; 
	if(!data::Load("train_data_3_classes.csv",trainX1)) {
		BOOST_FAIL("Could not load the train data train_data_3_classes.csv");
	}

	SetInputParam("training",std::move(trainX1)); //last column of trainX1 contains the class labels

	//training model using first training dataset
	mlpackMain();

	PerceptronModel model = CLI::GetParam<PerceptronModel>("output_model");

	resetSettings();

	arma::mat trainX2;

	if(!data::Load("train_data_5_classes.csv",trainX2)) {
		BOOST_FAIL("Could not load the train data train_data_5_classes.csv");
	}

	SetInputParam("training",std::move(trainX2)); //last column of trainX2 contains the class labels
	SetInputParam("input_model",std::move(model));

	Log::Fatal.ignoreInput=true;
	BOOST_REQUIRE_THROW(mlpackMain(),std::runtime_error);
	Log::Fatal.ignoreInput=false;
}

/**
  * Ensuring that the response size is checked
 **/
BOOST_AUTO_TEST_CASE(PerceptronWrongResponseSizeTest) 
{
	constexpr int D = 2;
	constexpr int N = 10;
	
	arma::mat trainX = arma::randu<arma::mat>(D,N);
	arma::Row<size_t> trainY; // response vector with wrong size

	trainY << 0 << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 1 << 0 << endr; // 8 responses

	SetInputParam("training", std::move(trainX));
	SetInputParam("labels", std::move(trainY));

	Log::Fatal.ignoreInput = true;
	BOOST_REQUIRE_THROW(mlpackMain(),std::runtime_error);
	Log::Fatal.ignoreInput = false;
}

/**
 * Ensuring that absence of responses is checked.
 */
BOOST_AUTO_TEST_CASE(PerceptronNoResponsesTest)
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
 * Ensuring that absence of training data is checked.
 */
BOOST_AUTO_TEST_CASE(PerceptronNoTrainingDataTest)
{
  arma::Row<size_t> trainY;
  trainY << 1 << 1 << 0 << 1 << 0 << 0 <<endr;

  SetInputParam("labels", std::move(trainY));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that model can saved / loaded and used. Ensuring that results are the
 * same.
 */
BOOST_AUTO_TEST_CASE(PerceptronModelReload)
{
  constexpr int N = 10;
  constexpr int D = 4;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr; // 10 responses
  
  arma::mat testX = arma::randu<arma::mat>(D, M);

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("test", testX);

  //first solution
  mlpackMain();

  PerceptronModel model = CLI::GetParam<PerceptronModel>("output_model");
  const arma::Row<size_t> testY1 = CLI::GetParam<arma::Row<size_t>>("output");

  resetSettings();

  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  //second solution
  mlpackMain();

  const arma::Row<size_t> testY2 = CLI::GetParam<arma::Row<size_t>>("output");

  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(), testY2.begin(), testY2.end());
}

/**
 * Ensuring that test data dimensionality is checked when model is loaded.
 */
BOOST_AUTO_TEST_CASE(PerceptronWrongDimOfTestData2)
{
  constexpr int N = 10;
  constexpr int D = 3;
  constexpr int M = 15;

  arma::mat trainX = arma::randu<arma::mat>(D, N);
  arma::Row<size_t> trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr; // 10 responses

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));

  // training the model
  mlpackMain();

  PerceptronModel model = CLI::GetParam<PerceptronModel>("output_model");

  resetSettings();

  arma::mat testX = arma::randu<arma::mat>(D - 1, M); // test data with Wrong dimensionality.
  SetInputParam("input_model", std::move(model));
  SetInputParam("test", std::move(testX));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Checking two options of specifying responses (extra row in train matrix and
 * extra parameter) and ensuring that predictions are the same.
 */
BOOST_AUTO_TEST_CASE(PerceptronResponsesRepresentationTest)
{
  arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}, {0,1,1}});
  arma::mat testX({{4.0,5.0},{1.0,6.0}});

  SetInputParam("training", trainX1);
  SetInputParam("test", testX);

  // The first solution.
  mlpackMain();

  const arma::Row<size_t> testY1 = CLI::GetParam<arma::Row<size_t>>("output");

  resetSettings();

  arma::mat trainX2({{1.0, 2.0, 3.0},{1.0, 4.0, 9.0}});
  arma::Row<size_t> trainY2({0,1,1});

  SetInputParam("training", std::move(trainX2));
  SetInputParam("labels", std::move(trainY2));
  SetInputParam("test", std::move(testX));

  // The second solution.
  mlpackMain();

  const arma::Row<size_t> testY2 = CLI::GetParam<arma::Row<size_t>>("output");

  BOOST_REQUIRE_EQUAL_COLLECTIONS(testY1.begin(), testY1.end(), testY2.begin(), testY2.end());
}

/**
 * Checking that that size and dimensionality of prediction is correct.
 */
BOOST_AUTO_TEST_CASE(PerceptronPredictionDimTest)
{
	constexpr int N = 10;
	constexpr int D = 3;

	arma::mat trainX = arma::randu<arma::mat>(D,N);
	arma::Row<size_t> trainY;

	trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr; // 10 responses

	arma::mat testX = arma::randu<arma::mat>(D,N);

	SetInputParam("training", std::move(trainX));
	SetInputParam("labels", std::move(trainY));
	SetInputParam("test", std::move(testX));

	mlpackMain();

	const arma::Row<size_t> testY = CLI::GetParam<arma::Row<size_t>>("output");

	BOOST_REQUIRE_EQUAL(testY.n_rows,1);
	BOOST_REQUIRE_EQUAL(testY.n_cols,N);
}

/**
  * Ensuring that the max_iterations is non negative
 **/ 
BOOST_AUTO_TEST_CASE(PerceptronNonNegMaxIterationTest) 
{
  constexpr int N = 10;
  constexpr int D = 3;

  arma::mat trainX = arma::randu<arma::mat>(D,N);
  arma::Row<size_t> trainY;

  trainY << 0 << 1 << 0 << 1 << 1 << 1 << 0 << 1 << 0 << 0 << endr; // 10 responses

  SetInputParam("training", std::move(trainX));
  SetInputParam("labels", std::move(trainY));
  SetInputParam("max_iterations", int (-1));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
BOOST_AUTO_TEST_SUITE_END();

