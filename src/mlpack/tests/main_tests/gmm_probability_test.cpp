/**
 * @file gmm_probability_test.cpp
 * @author Gaurav Tripathi
 *
 * Test mlpackMain() of gmm_probability_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
              
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "GmmProbability"

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/gmm/gmm_probabiity_main.cpp>
#include <mlpack/methods/gmm/gmm_train_main.cpp>

#include "test_helper.hpp"

#include <boost/test/unit_tests.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct GmmProbabilityTestFixture
{
  public:
   GmmProbabilityTestFixture()
   {
     CLI::RestoreSettings(testname);
   }

   ~GmmProbabilityTestFixture()
   {
     bindings::tests::CleanMemory();
     CLI::ClearSettings();
   }
};

BOOST_FIXTURE_TEST_SUITE(GmmProbabilityMainTest,GmmProbabilityTestFixture);

// Make sure that input model is provided.
BOOST_AUTO_TEST_CASE(GmmProbabilityInputModelTest)
{
  arma::mat inputData;

  InputData << 0 << 1 << 2 << 3 << 4 << 5 << 6 << 7 << endr
            << 0 << 4 << 3 << 4 << 8 << 9 << 2 << 5 << endr;

  SetInputParam("input" , std::move(inputData));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}
  
// Make sure that input points are provided.
BOOST_AUTO_TEST_CASE(GmmProbabilityInputPoints)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");    
  
  GMM gmm(1, 2);
  gmm.Train(inputData, 2);

  SetInputParam("input_model" , std::move(gmm));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;     
}

// Checking the input and output dimensionality.
BOOST_AUTO_TEST_CASE(GmmProbabilityDimensionality)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");
  
  GMM gmm(1, 2);
  gmm.Train(inputData, 10);
  
  arma::mat InputData;
  InputData << 0 << 1 << 2<< 3 << 4 << 5 << 6 << 7 << endr
            << 0 << 4 << 3 << 4 << 8 << 9 << 2 << 5 << endr;

  SetInputParam("input_model", std::move(gmm));
  SetInputParam("input", std::move(inputData));

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols,8);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows,2);
}

BOOST_AUTO_TEST_SUITE_END();                                