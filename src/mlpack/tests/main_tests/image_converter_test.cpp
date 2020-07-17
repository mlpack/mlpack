/**
 * @file tests/main_tests/image_converter_test.cpp
 * @author Jeffin Sam
 *
 * Test mlpackMain() of load_save_image_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
static const std::string testName = "ImageConverter";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/image_converter_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct ImageConverterTestFixture
{
 public:
  ImageConverterTestFixture()
  {
    // Cache in the options for this program.
    IO::RestoreSettings(testName);
  }

  ~ImageConverterTestFixture()
  {
    // Clear the settings.
    remove("test_image777.png");
    remove("test_image999.png");
    bindings::tests::CleanMemory();
    IO::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(ImageConverterMainTest,
                         ImageConverterTestFixture);

BOOST_AUTO_TEST_CASE(LoadImageTest)
{
  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});

  mlpackMain();
  arma::mat output = IO::GetParam<arma::mat>("output");
  // width * height * channels.
  BOOST_REQUIRE_EQUAL(output.n_rows, 50 * 50 * 3);
  BOOST_REQUIRE_EQUAL(output.n_cols, 2);
}

BOOST_AUTO_TEST_CASE(SaveImageTest)
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
  mlpackMain();

  IO::ClearSettings();
  IO::RestoreSettings(testName);

  SetInputParam<vector<string>>("input", {"test_image777.png",
    "test_image999.png"});
  SetInputParam("height", 5);
  SetInputParam("width", 5);
  SetInputParam("channels", 3);

  mlpackMain();
  arma::mat output = IO::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(output.n_rows, 5 * 5 * 3);
  BOOST_REQUIRE_EQUAL(output.n_cols, 2);
  for (size_t i = 0; i < output.n_elem; ++i)
    BOOST_REQUIRE_CLOSE(testimage[i], output[i], 1e-5);
}

/**
 * Check whether binding throws error if height, width or channel are not
 * specified.
 */
BOOST_AUTO_TEST_CASE(IncompleteTest)
{
  arma::mat testimage = arma::conv_to<arma::mat>::from(
      arma::randi<arma::Mat<unsigned char>>((5 * 5 * 3), 2));
  SetInputParam<vector<string>>("input", {"test_image777.png",
      "test_image999.png"});
  SetInputParam("save", true);
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("dataset", testimage);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check for invalid height values.
 */
BOOST_AUTO_TEST_CASE(InvalidInputTest)
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

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check for invalid width values.
 */
BOOST_AUTO_TEST_CASE(InvalidWidthTest)
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

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check for invalid channel values.
 */
BOOST_AUTO_TEST_CASE(InvalidChannelTest)
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

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check for invalid input values.
 */
BOOST_AUTO_TEST_CASE(EmptyInputTest)
{
  SetInputParam<vector<string>>("input", {});
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("channels", 50);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
