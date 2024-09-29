/**
 * @file tests/main_tests/preprocess_scale_test.cpp
 * @author Jeffin Sam
 *
 * Test RUN_BINDING() of preprocess_scale_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/preprocess/preprocess_scale_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(PreprocessScaleTestFixture);

arma::mat scaleMainDataset = "-1 -0.5 0 1;"
                             "2 6 10 18;";

/**
 * Check that two different scalers give two different output.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "TwoScalerTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  // Input custom data points.
  std::string method = "max_abs_scaler";
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);

  RUN_BINDING();
  arma::mat maxAbsScalerOutput = params.Get<arma::mat>("output");

  CleanMemory();
  ResetSettings();

  method = "standard_scaler";
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));

  RUN_BINDING();
  arma::mat standardScalerOutput = params.Get<arma::mat>("output");

  CheckMatricesNotEqual(standardScalerOutput, maxAbsScalerOutput);
}

/**
 * Check that two different option for a particular scaler give two
 * different output.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "TwoOptionTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "min_max_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");

  CleanMemory();
  ResetSettings();

  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);

  RUN_BINDING();
  arma::mat outputWithParam = params.Get<arma::mat>("output");

  CheckMatricesNotEqual(output, outputWithParam);
}

/**
 * Check that passing unrelated option don't change anything.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "UnrelatedOptionTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "standard_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);

  RUN_BINDING();
  arma::mat scaled = params.Get<arma::mat>("output");

  CleanMemory();
  ResetSettings();

  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);
  SetInputParam("epsilon", 0.005);

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");

  CheckMatrices(scaled, output);
}

/**
 * Check Inverse Scaling is working.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "InverseScalingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "zca_whitening";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));

  RUN_BINDING();
  arma::mat scaled = params.Get<arma::mat>("output");

  SetInputParam("input", scaled);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");
  CheckMatrices(scaleMainDataset, output);
}

/**
 * Check Saved model is working.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "SavedModelTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));

  RUN_BINDING();
  arma::mat scaled = params.Get<arma::mat>("output");

  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");
  CheckMatrices(scaled, output);
}

/**
 * Check different epsilon for PCA give two different output.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "EpsilonTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);

  RUN_BINDING();
  arma::mat scaled = params.Get<arma::mat>("output");

  CleanMemory();
  ResetSettings();

  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("epsilon", 1.0);

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");

  CheckMatricesNotEqual(scaled, output);
}

/**
 * Check for invalid epsilon.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "InvalidEpsilonTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("epsilon", -1.0);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for invalid range in min_max_scaler.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "InvalidRangeTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "min_max_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 4);
  SetInputParam("max_value", 2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for invalid scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "InvalidScalerTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "invalid_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("min_value", 4);
  SetInputParam("max_value", 2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for Standard scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "StandardScalerBindingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "standard_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);
  REQUIRE_NOTHROW(RUN_BINDING());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  REQUIRE_NOTHROW(RUN_BINDING());
}

/**
 * Check for MaxAbs scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "MaxAbsScalerBindingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "max_abs_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);
  REQUIRE_NOTHROW(RUN_BINDING());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  REQUIRE_NOTHROW(RUN_BINDING());
}

/**
 * Check for MinMax scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "MinMaxScalerBindingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "min_max_scaler";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);
  SetInputParam("min_value", 2);
  SetInputParam("max_value", 4);
  REQUIRE_NOTHROW(RUN_BINDING());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  REQUIRE_NOTHROW(RUN_BINDING());
}

/**
 * Check for PCA scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "PCAScalerBindingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "pca_whitening";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);
  SetInputParam("epsilon", 1.0);
  REQUIRE_NOTHROW(RUN_BINDING());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  REQUIRE_NOTHROW(RUN_BINDING());
}

/**
 * Check for ZCA scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "ZCAScalerBindingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "zca_whitening";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);
  SetInputParam("epsilon", 1.0);
  REQUIRE_NOTHROW(RUN_BINDING());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  REQUIRE_NOTHROW(RUN_BINDING());
}

/**
 * Check for Mean Normalization scaler type.
 */
TEST_CASE_METHOD(PreprocessScaleTestFixture, "MeanNormalizationBindingTest",
                 "[PreprocessScaleMainTest][BindingTests]")
{
  std::string method = "mean_normalization";
  // Input custom data points.
  SetInputParam("input", scaleMainDataset);
  SetInputParam("scaler_method", method);
  REQUIRE_NOTHROW(RUN_BINDING());
  SetInputParam("scaler_method", std::move(method));
  SetInputParam("input", scaleMainDataset);
  SetInputParam("input_model",
                params.Get<ScalingModel*>("output_model"));
  SetInputParam("inverse_scaling", true);
  REQUIRE_NOTHROW(RUN_BINDING());
}
