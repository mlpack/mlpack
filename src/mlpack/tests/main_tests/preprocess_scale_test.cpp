/**
 * @file preprocess_scale_test.cpp
 * @author Jeffin Sam
 *
 * Test mlpackMain() of preprocess_scale_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "PreprocessScale";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/preprocess_scale_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

#include <cmath>

using namespace mlpack;

struct PreprocessScaleTestFixture
{
 public:
  PreprocessScaleTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~PreprocessScaleTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(PreprocessScaleMainTest,
                         PreprocessScaleTestFixture);

arma::mat dataset = "-1 -0.5 0 1;"
                    "2 6 10 18;";
/**
 * Check that two different scalers give two different output.
 */
BOOST_AUTO_TEST_CASE(TwoScalerTest)
{
  std::string method = "max_abs_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));

  mlpackMain();
  arma::mat max_abs_scaler_output = CLI::GetParam<arma::mat>("output");

  method = "standard_scaler"; 
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));

  mlpackMain();
  arma::mat standard_scaler_output = CLI::GetParam<arma::mat>("output");

  CheckMatricesNotEqual(standard_scaler_output, max_abs_scaler_output);
}

/**
 * Check that two different option for a particular scaler give two
 * different output.
 */
BOOST_AUTO_TEST_CASE(TwoOptionTest)
{
  std::string method = "min_max_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);

  mlpackMain();
  arma::mat output = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);

  mlpackMain();
  arma::mat output_with_param = CLI::GetParam<arma::mat>("output");

  CheckMatricesNotEqual(output, output_with_param);
}

/**
 * Check that passing unrelated option don't change anything.
 */
BOOST_AUTO_TEST_CASE(UnrelatedOptionTest)
{
  std::string method = "standard_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);
  SetInputParam("epsilon", 0.005);
  mlpackMain();
  arma::mat output = CLI::GetParam<arma::mat>("output");
  arma::mat scaled = "-1.18321596 -0.50709255  0.16903085 1.52127766;"
                   "-1.18321596 -0.50709255  0.16903085 1.52127766;";
  CheckMatrices(scaled, output);
}

BOOST_AUTO_TEST_SUITE_END();
