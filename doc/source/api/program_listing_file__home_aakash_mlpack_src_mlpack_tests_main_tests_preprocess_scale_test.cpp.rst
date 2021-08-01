
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_preprocess_scale_test.cpp:

Program Listing for File preprocess_scale_test.cpp
==================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_preprocess_scale_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/preprocess_scale_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   #include <mlpack/core.hpp>
   static const std::string testName = "PreprocessScale";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/preprocess/preprocess_scale_main.cpp>
   
   #include "test_helper.hpp"
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct PreprocessScaleTestFixture
   {
    public:
     static arma::mat dataset;
     PreprocessScaleTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~PreprocessScaleTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   arma::mat PreprocessScaleTestFixture::dataset = "-1 -0.5 0 1;"
                                                   "2 6 10 18;";
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "TwoScalerTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     // Input custom data points.
     std::string method = "max_abs_scaler";
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
   
     mlpackMain();
     arma::mat maxAbsScalerOutput = IO::GetParam<arma::mat>("output");
   
     bindings::tests::CleanMemory();
   
     method = "standard_scaler";
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
   
     mlpackMain();
     arma::mat standardScalerOutput = IO::GetParam<arma::mat>("output");
   
     CheckMatricesNotEqual(standardScalerOutput, maxAbsScalerOutput);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "TwoOptionTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "min_max_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
   
     mlpackMain();
     arma::mat output = IO::GetParam<arma::mat>("output");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("min_value", 2);
     SetInputParam("max_value", 4);
   
     mlpackMain();
     arma::mat output_with_param = IO::GetParam<arma::mat>("output");
   
     CheckMatricesNotEqual(output, output_with_param);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "UnrelatedOptionTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "standard_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
   
     mlpackMain();
     arma::mat scaled = IO::GetParam<arma::mat>("output");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("min_value", 2);
     SetInputParam("max_value", 4);
     SetInputParam("epsilon", 0.005);
   
     mlpackMain();
     arma::mat output = IO::GetParam<arma::mat>("output");
   
     CheckMatrices(scaled, output);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "InverseScalingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "zca_whitening";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
   
     mlpackMain();
     arma::mat scaled = IO::GetParam<arma::mat>("output");
   
     SetInputParam("input", scaled);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
   
     mlpackMain();
     arma::mat output = IO::GetParam<arma::mat>("output");
     CheckMatrices(dataset, output);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "SavedModelTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "pca_whitening";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
   
     mlpackMain();
     arma::mat scaled = IO::GetParam<arma::mat>("output");
   
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
   
     mlpackMain();
     arma::mat output = IO::GetParam<arma::mat>("output");
     CheckMatrices(scaled, output);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "EpsilonTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "pca_whitening";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
   
     mlpackMain();
     arma::mat scaled = IO::GetParam<arma::mat>("output");
   
     bindings::tests::CleanMemory();
   
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("epsilon", 1.0);
   
     mlpackMain();
     arma::mat output = IO::GetParam<arma::mat>("output");
   
     CheckMatricesNotEqual(scaled, output);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "InvalidEpsilonTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "pca_whitening";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("epsilon", -1.0);
   
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "InvalidRangeTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "min_max_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("min_value", 4);
     SetInputParam("max_value", 2);
   
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "InvalidScalerTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "invalid_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("min_value", 4);
     SetInputParam("max_value", 2);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "StandardScalerBindingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "standard_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
     REQUIRE_NOTHROW(mlpackMain());
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
     REQUIRE_NOTHROW(mlpackMain());
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "MaxAbsScalerBindingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "max_abs_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
     REQUIRE_NOTHROW(mlpackMain());
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
     REQUIRE_NOTHROW(mlpackMain());
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "MinMaxScalerBindingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "min_max_scaler";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
     SetInputParam("min_value", 2);
     SetInputParam("max_value", 4);
     REQUIRE_NOTHROW(mlpackMain());
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
     REQUIRE_NOTHROW(mlpackMain());
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "PCAScalerBindingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "pca_whitening";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
     SetInputParam("epsilon", 1.0);
     REQUIRE_NOTHROW(mlpackMain());
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
     REQUIRE_NOTHROW(mlpackMain());
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "ZCAScalerBindingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "zca_whitening";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
     SetInputParam("epsilon", 1.0);
     REQUIRE_NOTHROW(mlpackMain());
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
     REQUIRE_NOTHROW(mlpackMain());
   }
   
   TEST_CASE_METHOD(PreprocessScaleTestFixture, "MeanNormalizationBindingTest",
                    "[PreprocessScaleMainTest][BindingTests]")
   {
     std::string method = "mean_normalization";
     // Input custom data points.
     SetInputParam("input", dataset);
     SetInputParam("scaler_method", method);
     REQUIRE_NOTHROW(mlpackMain());
     SetInputParam("scaler_method", std::move(method));
     SetInputParam("input", dataset);
     SetInputParam("input_model",
                   IO::GetParam<ScalingModel*>("output_model"));
     SetInputParam("inverse_scaling", true);
     REQUIRE_NOTHROW(mlpackMain());
   }
