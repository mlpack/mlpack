/**
 * @file load_save_image_test.cpp
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
static const std::string testName = "LoadSaveImage";

#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/preprocess/load_save_image_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct LoadSaveImageTestFixture
{
 public:
  LoadSaveImageTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~LoadSaveImageTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(LoadSaveImageMainTest,
                         LoadSaveImageTestFixture);


/**
 * Check that two different scalers give two different output.
 */

// arma::mat testimage = arma::conv_to<arma::mat>::from(
//     arma::randi<arma::Mat<unsigned char>>((50 * 50 * 3), 2));

// Global variable to avoid creating of multiple files
arma::mat input;
arma::mat output;

BOOST_AUTO_TEST_CASE(LoadImageTest)
{
  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("channel", 3);

  mlpackMain();
  output = CLI::GetParam<arma::mat>("output");
  BOOST_REQUIRE_EQUAL(output.n_rows, 50 * 50 * 3); // width * height * channels.
  BOOST_REQUIRE_EQUAL(output.n_cols, 2);

  // SetInputParam<vector<string>>("input", {"test_image777.png", "test_image999.png"});
  // SetInputParam("height", 50);
  // SetInputParam("width", 50);
  // SetInputParam("channel", 3);
  // SetInputParam("save", true);
  // SetInputParam("dataset", output);
  // mlpackMain();
  // std::cout<<"output given is "<<output.n_rows<<"\n";
  // bindings::tests::CleanMemory();
  // SetInputParam<vector<string>>("input", {"test_image777.png", "test_image999.png"});
  // SetInputParam("height", 50);
  // SetInputParam("width", 50);
  // SetInputParam("channel", 3);

  // mlpackMain();
  // input = CLI::GetParam<arma::mat>("output");
  //  std::cout<<"outputnew is "<<input.n_rows<<"\n";
  // BOOST_REQUIRE_EQUAL(input.n_rows, 50 * 50 * 3); // width * height * channels.
  // BOOST_REQUIRE_EQUAL(input.n_cols, 2);
  // for (size_t i = 0; i < output.n_elem; ++i)
  //   BOOST_REQUIRE_CLOSE(input[i], output[i], 1e-5);
}

BOOST_AUTO_TEST_CASE(SaveImageTest)
{
  // std::cout<<"output entry is "<<output.n_rows<<"\n";

  // SetInputParam<vector<string>>("input", {"test_image777.png", "test_image999.png"});
  // SetInputParam("height", 50);
  // SetInputParam("width", 50);
  // SetInputParam("channel", 3);
  // SetInputParam("save", true);
  // SetInputParam("dataset", output);
  // mlpackMain();
  // std::cout<<"output given is "<<output.n_rows<<"\n";

  // SetInputParam<vector<string>>("input", {"test_image777.png", "test_image999.png"});
  // SetInputParam("height", 50);
  // SetInputParam("width", 50);
  // SetInputParam("channel", 3);

  // mlpackMain();
  // input = CLI::GetParam<arma::mat>("output");
  //  std::cout<<"outputnew is "<<input.n_rows<<"\n";
  // BOOST_REQUIRE_EQUAL(input.n_rows, 50 * 50 * 3); // width * height * channels.
  // BOOST_REQUIRE_EQUAL(input.n_cols, 2);
  // for (size_t i = 0; i < output.n_elem; ++i)
  //   BOOST_REQUIRE_CLOSE(input[i], output[i], 1e-5);

}

/**
 * Check Saved model is working.
 */
BOOST_AUTO_TEST_CASE(SavedModelTest)
{
  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("channel", 3);

  mlpackMain();
  arma::mat randomOutput = CLI::GetParam<arma::mat>("output");

  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});
  SetInputParam("input_model",
                CLI::GetParam<ImageInfo*>("output_model"));

  mlpackMain();
  arma::mat savedOutput = CLI::GetParam<arma::mat>("output");
  CheckMatrices(randomOutput, savedOutput);
}

/**
 * Check transpose option give two different output.
 */
BOOST_AUTO_TEST_CASE(TransposeTest)
{
  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});
  SetInputParam("height", 50);
  SetInputParam("width", 50);
  SetInputParam("channel", 3);

  mlpackMain();
  arma::mat normalOutput = CLI::GetParam<arma::mat>("output");

  SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});
  SetInputParam("input_model",
                CLI::GetParam<ImageInfo*>("output_model"));
  SetInputParam("transpose", true);
  mlpackMain();
  arma::mat transposeOutput = CLI::GetParam<arma::mat>("output");

  CheckMatricesNotEqual(normalOutput, transposeOutput);
}

BOOST_AUTO_TEST_SUITE_END();
