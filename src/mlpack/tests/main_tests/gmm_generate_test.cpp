/**
 * @file gmm_generate_test.cpp
 * @author Gaurav Tripathi
 *
 * Test mlpackMain() of gmm_generate_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "gmm_generate"

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/gmm/gmm_generate_main.cpp>
#include <mlpack/methods/gmm/gmm_train_main.cpp>
#include <mlpack/methods/gmm/no_constraint.hpp>
#include <mlpack/methods/gmm/positive_definite_constraint.hpp>
#include <mlpack/methods/gmm/diagonal_constraint.hpp>
#include <mlpack/methods/gmm/eigenvalue_ratio_constraint.hpp>
#include "test_helper.hpp"

#include <boost/test/unit_tests.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct GmmGenerateTestFixture
{
  public:
   GmmGenerateTestFixture()
   {
     CLI::RestoreSettings(testname);
   }

   ~GmmGenerateTestFixture()
   {
     bindings::tests::CleanMemory();
     CLI::ClearSettings();
   }
};

BOOST_FIXTURE_TEST_SUITE(GmmGenerateMainTest,
                         GmmGenerateTestFixture);

// Making sure samples are provided.
BOOST_AUTO_TEST_CASE(GmmGenerateSamples)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");
  
  GMM gmm(1, 2);
  gmm.Train(inputData, 10);
  SetInputParam("input_model", std::move(gmm));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Checking dimensionality of output.
BOOST_AUTO_TEST_CASE(GmmGenerateDimensionality)
{ int samples =10;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");
  
  GMM gmm(1, 2);
  gmm.Train(data, 10);

  SetInputParam("input_model", std::move(gmm));
  SetInputParam("samples", samples);

  mlpackMain();

  arma::mat output = CLI::GetParam<arma::mat>("output");

  BOOST_REQUIRE_EQUAL(output.n_rows, gmm->Dimensionality());
  BOOST_REQUIRE_EQUAL(output.n_cols, samples);
 
}

// Samples passed should be positive.
BOOST_AUTO_TEST_CASE(GmmGeneratePositiveSample)
{
  int samples = -1;

  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
  BOOST_FAIL("Unable to load train dataset vc2.csv!");
  
  GMM gmm(1, 2);
  gmm.Train(data, 10);

  SetInputParam("input_model", std::move(gmm));
  SetInputParam("samples", samples); // Invalid


  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();