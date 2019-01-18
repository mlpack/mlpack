#include<string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std:: string testname = "gmm_train"


#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/gmm/gmm_train_main.cpp>
#include <mlpack/methods/kmeans/kmeans_main.cpp>

#include "no_constraint.hpp"
#include "diagonal_constraint.hpp"

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct GmmTrainTestFixture
{
public:
  GmmTrainTestFixture()
    {
  CLI::RestoreSettings(testname);
}

  ~GmmTrainTestFixture()
{
  CLI::ClearSettings();
}
};

void ResetGmmTrainSetting()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testname);
}

BOOST_FIXTURE_TEST_SUITE(GmmTrainMainTest , GmmTrainTestFixture);

//To check if the gaussian is positive or not
BOOST_AUTO_TEST_CASE(GmmTrainValidGaussian)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");

    
  SetInputParam("input" , std::move(inputdata));
  SetInputParam("gaussian" , (int) -1); //invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

}

//Number of trials is provided or not
BOOST_AUTO_TEST_CASE(GmmTrainValidTrials)
{
  int g = 3;
  int trials = 0;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");


  SetInputParam("input" , std::move(inputdata));
  SetInputParam("gaussians" , g);
  SetInputParam("trials" , trials) //invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
// if covariance flag is false then no_force_positive parameter is not specified
BOOST_AUTO_TEST_CASE(GmmTrainDiagonalCovarianceCheck)
{
  int g = 3;
  int t = 2;
  int n_f_p = 2;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input" , std::move(inputdata));
  SetInputParam("gaussians" , g);
  SetInputParam("trials" , t);
  SetInputParam("Diagonal_covariance" , false);
  SetinputParam("no_force_positive" , n_f_p); //invalid
  SetInputParam("noise" , (int) 0);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain() , runtime_error);
  Log::Fatal.ignoreInput = false;

}


//max iterations must be positive)
BOOST_AUTO_TEST_CASE(GmmTrainMaxIterations)
{
  int g = 3;
  int t = 2;
  int mi = -1;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");


  SetinputParam("input" , std::move(inputdata));
  SetinputParam("gaussians", g);
  SetinputParam("trials" , t);
  SetinputParam("max_iterations" ,mi)//invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain() , runtime_error);
  Log::Fatal.ignoreInput = false;
}

//tolerence must be positive 
BOOST_AUTO_TEST_CASE(GmmTrainTolerence)
{
  int g = 3;
  int t = 2;
  int mi = 3;
  int tol = -1;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetinputParam("input" , std::move(inputdata));
  SetinputParam("gaussians", g);
  SetinputParam("trials" , t);
  SetinputParam("max_iterations" ,mi);
  SetinputParam("tolerence" , tol);//invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain() , runtime_error);
  Log::Fatal.ignoreInput = false;

}
 
 /**
 * Checking that percentage is between 0 and 1 when --refined_start is specified
*/
BOOST_AUTO_TEST_CASE(RefinedStartPercentageTest)
{
  int g = 3;
  int t = 2;
  int mi = 3;
  int tol = -1;
  int c = 2;
  double P = 2.0;
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("clusters", c);
  SetInputParam("percentage", std::move(P));     // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}




BOOST_AUTO_TEST_SUITE_END();

