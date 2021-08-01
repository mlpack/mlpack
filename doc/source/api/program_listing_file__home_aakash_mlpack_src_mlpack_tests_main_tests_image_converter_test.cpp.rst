
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_image_converter_test.cpp:

Program Listing for File image_converter_test.cpp
=================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_image_converter_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/image_converter_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   #include <mlpack/core.hpp>
   static const std::string testName = "ImageConverter";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/preprocess/image_converter_main.cpp>
   
   #include "test_helper.hpp"
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   
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
   
   TEST_CASE_METHOD(ImageConverterTestFixture, "LoadImageTest",
                    "[ImageConverterMainTest][BindingTests]")
   {
     SetInputParam<vector<string>>("input", {"test_image.png", "test_image.png"});
   
     mlpackMain();
     arma::mat output = IO::GetParam<arma::mat>("output");
     // width * height * channels.
     REQUIRE(output.n_rows == 50 * 50 * 3);
     REQUIRE(output.n_cols == 2);
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
     REQUIRE(output.n_rows == 5 * 5 * 3);
     REQUIRE(output.n_cols == 2);
     for (size_t i = 0; i < output.n_elem; ++i)
       REQUIRE(testimage[i] == Approx(output[i]).epsilon(1e-7));
   }
   
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
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
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
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
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
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
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
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(ImageConverterTestFixture, "EmptyInputTest",
                    "[ImageConverterMainTest][BindingTests]")
   {
     SetInputParam<vector<string>>("input", {});
     SetInputParam("height", 50);
     SetInputParam("width", 50);
     SetInputParam("channels", 50);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
