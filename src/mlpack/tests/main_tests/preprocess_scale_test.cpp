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
  // Input custom data points.
  std::string method = "max_abs_scaler";
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);

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
  SetInputParam("scaler_method", method);

  mlpackMain();
  arma::mat scaled = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);
  SetInputParam("epsilon", 0.005);

  mlpackMain();
  arma::mat output = CLI::GetParam<arma::mat>("output");

  CheckMatrices(scaled, output);
}

/**
 * Check Inverse Scaling is working.
 */
BOOST_AUTO_TEST_CASE(InverseScalingTest)
{
  std::string method = "zca_whitening";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));

  mlpackMain();
  arma::mat scaled = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", scaled);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);

  mlpackMain();
  arma::mat output = CLI::GetParam<arma::mat>("output");
  CheckMatrices(dataset, output);
}

/**
 * Check Saved model is working.
 */
BOOST_AUTO_TEST_CASE(SavedModelTest)
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));

  mlpackMain();
  arma::mat scaled = CLI::GetParam<arma::mat>("output");

  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));

  mlpackMain();
  arma::mat output = CLI::GetParam<arma::mat>("output");
  CheckMatrices(scaled, output);
}

/**
 * Check different epsilon for PCA give two different output.
 */
BOOST_AUTO_TEST_CASE(EpsilonTest)
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);

  mlpackMain();
  arma::mat scaled = CLI::GetParam<arma::mat>("output");

  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("epsilon", 1.0);

  mlpackMain();
  arma::mat output = CLI::GetParam<arma::mat>("output");

  CheckMatricesNotEqual(scaled, output);
}

/**
 * Check for invalid epsilon.
 */
BOOST_AUTO_TEST_CASE(InvalidEpsilonTest)
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("epsilon", -1.0);

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
}

/**
 * Check for invalid range in min_max_scaler.
 */
BOOST_AUTO_TEST_CASE(InvalidRangeTest)
{
  std::string method = "min_max_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 4);
  SetInputParam("max_value", 2);

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
}

/**
 * Check for invalid scaler type.
 */
BOOST_AUTO_TEST_CASE(InvalidScalerTest)
{
  std::string method = "invalid_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 4);
  SetInputParam("max_value", 2);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check for Standard scaler type.
 */
BOOST_AUTO_TEST_CASE(StandardScalerTest)
{
  std::string method = "standard_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
}

/**
 * Check for MaxAbs scaler type.
 */
BOOST_AUTO_TEST_CASE(MaxAbsScalerTest)
{
  std::string method = "max_abs_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
}

/**
 * Check for MinMax scaler type.
 */
BOOST_AUTO_TEST_CASE(MinMaxScalerTest)
{
  std::string method = "min_max_scaler";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
}

/**
 * Check for PCA scaler type.
 */
BOOST_AUTO_TEST_CASE(PCAScalerTest)
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);
  SetInputParam("epsilon", 1.0);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
}

/**
 * Check for ZCA scaler type.
 */
BOOST_AUTO_TEST_CASE(ZCAScalerTest)
{
  std::string method = "zca_whitening";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);
  SetInputParam("epsilon", 1.0);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
}

/**
 * Check for Mean Normalization scaler type.
 */
BOOST_AUTO_TEST_CASE(MeanNormalizationTest)
{
  std::string method = "mean_normalization";
  // Input custom data points.
  SetInputParam("input", dataset);
  SetInputParam("scaler_method", method);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", dataset);
  SetInputParam("input_model",
                CLI::GetParam<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  BOOST_REQUIRE_NO_THROW(mlpackMain());
}

BOOST_AUTO_TEST_SUITE_END();
