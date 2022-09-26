/**
 * @file tests/main_tests/image_converter_test.cpp
 * @author Jeffin Sam
 *
 * Test RUN_BINDING() of load_save_image_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/preprocess/image_converter_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>

#include "main_test_fixture.hpp"

#include "../test_catch_tools.hpp"
#include "../catch.hpp"

using namespace mlpack;

#ifdef HAS_STB // Compile this only if stb is present.

BINDING_TEST_FIXTURE(ImageConverterTestFixture);

TEST_CASE_METHOD(ImageConverterTestFixture, "LoadImageTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");
  // width * height * channels.
  REQUIRE(output.n_rows == 50 * 50 * 3);
  REQUIRE(output.n_cols == 2);

  remove("test_image777.png");
  remove("test_image999.png");
}

TEST_CASE_METHOD(ImageConverterTestFixture, "SaveImageTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  arma::mat testimage = arma::conv_to<arma::mat>::from(
      arma::randi<arma::Mat<unsigned char>>((5 * 5 * 3), 2));
  SetInputParam<vector<string>>("input", {"test_image777.png",
      "test_image999.png"});
  SetInputParam("height", 5);
  SetInputParam("width", 5);
  SetInputParam("channels", 3);
  SetInputParam("save", true);
  SetInputParam("dataset", testimage);
  RUN_BINDING();

  ResetSettings();

  SetInputParam<vector<string>>("input", {"test_image777.png",
    "test_image999.png"});
  SetInputParam("height", 5);
  SetInputParam("width", 5);
  SetInputParam("channels", 3);

  RUN_BINDING();
  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(output.n_rows == 5 * 5 * 3);
  REQUIRE(output.n_cols == 2);
  for (size_t i = 0; i < output.n_elem; ++i)
    REQUIRE(testimage[i] == Approx(output[i]).epsilon(1e-7));

  remove("test_image777.png");
  remove("test_image999.png");
}

/**
 * Check whether binding throws error if height, width or channel are not
 * specified.
 */
TEST_CASE_METHOD(ImageConverterTestFixture, "IncompleteTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  arma::mat testimage = arma::conv_to<arma::mat>::from(
      arma::randi<arma::Mat<unsigned char>>((5 * 5 * 3), 2));
  SetInputParam<vector<string>>("input", {"test_image777.png",
      "test_image999.png"});
  SetInputParam("save", true);
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("dataset", testimage);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for invalid height values.
 */
TEST_CASE_METHOD(ImageConverterTestFixture, "InvalidInputTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  arma::mat testimage = arma::conv_to<arma::mat>::from(
      arma::randi<arma::Mat<unsigned char>>((5 * 5 * 3), 2));
  SetInputParam<vector<string>>("input", {"test_image777.png",
      "test_image999.png"});
  SetInputParam("save", true);
  SetInputParam("dataset", testimage);

  SetInputParam("height", -50);
  SetInputParam("width", 50);
  SetInputParam("channels", 3);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for invalid width values.
 */
TEST_CASE_METHOD(ImageConverterTestFixture, "InvalidWidthTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  arma::mat testimage = arma::conv_to<arma::mat>::from(
      arma::randi<arma::Mat<unsigned char>>((5 * 5 * 3), 2));
  SetInputParam<vector<string>>("input", {"test_image777.png",
      "test_image999.png"});
  SetInputParam("save", true);
  SetInputParam("dataset", testimage);
  SetInputParam("height", 50);
  SetInputParam("width", -50);
  SetInputParam("channels", 3);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for invalid channel values.
 */
TEST_CASE_METHOD(ImageConverterTestFixture, "InvalidChannelTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  arma::mat testimage = arma::conv_to<arma::mat>::from(
      arma::randi<arma::Mat<unsigned char>>((5 * 5 * 3), 2));
  SetInputParam<vector<string>>("input", {"test_image777.png",
      "test_image999.png"});
  SetInputParam("save", true);
  SetInputParam("dataset", testimage);
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("channels", -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check for invalid input values.
 */
TEST_CASE_METHOD(ImageConverterTestFixture, "EmptyInputTest",
                 "[ImageConverterMainTest][BindingTests]")
{
  SetInputParam<vector<string>>("input", {});
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("channels", 50);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

#endif // HAS_STB.
