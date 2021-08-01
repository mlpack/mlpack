
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_bayesian_linear_regression_test.cpp:

Program Listing for File bayesian_linear_regression_test.cpp
============================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_bayesian_linear_regression_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/bayesian_linear_regression_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "BayesianLinearRegression";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression_main.cpp>
   
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct BRTestFixture
   {
    public:
     BRTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~BRTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(BRTestFixture,
                    "BRCenter0Scale0",
                    "[BayesianLinearRegressionMainTest][BindingTests]")
   {
     int n = 50, m = 4;
     arma::mat matX = arma::randu<arma::mat>(m, n);
     arma::rowvec omega = arma::randu<arma::rowvec>(m);
     arma::rowvec y =  omega * matX;
   
     SetInputParam("input", std::move(matX));
     SetInputParam("responses", std::move(y));
     SetInputParam("center", false);
   
     mlpackMain();
   
     BayesianLinearRegression* estimator =
         IO::GetParam<BayesianLinearRegression*>("output_model");
   
     REQUIRE(estimator->DataOffset().n_elem == 0);
     REQUIRE(estimator->DataScale().n_elem == 0);
   }
   
   TEST_CASE_METHOD(BRTestFixture,
                    "BayesianLinearRegressionSavedEqualCode",
                    "[BayesianLinearRegressionMainTest][BindingTests]")
   {
     int n = 10, m = 4;
     arma::mat matX = arma::randu<arma::mat>(m, n);
     arma::mat matXtest = arma::randu<arma::mat>(m, 2 * n);
     const arma::rowvec omega = arma::randu<arma::rowvec>(m);
     arma::rowvec y =  omega * matX;
   
     BayesianLinearRegression model;
     model.Train(matX, y);
   
     arma::rowvec responses;
     model.Predict(matXtest, responses);
   
     SetInputParam("input", std::move(matX));
     SetInputParam("responses", std::move(y));
   
     mlpackMain();
   
     IO::GetSingleton().Parameters()["input"].wasPassed = false;
     IO::GetSingleton().Parameters()["responses"].wasPassed = false;
   
     SetInputParam("input_model",
                   IO::GetParam<BayesianLinearRegression*>("output_model"));
     SetInputParam("test", std::move(matXtest));
   
     mlpackMain();
   
     arma::mat ytest = std::move(responses);
     // Check that initial output and output using saved model are same.
     CheckMatrices(ytest, IO::GetParam<arma::mat>("predictions"));
   }
   
   TEST_CASE_METHOD(BRTestFixture,
                    "CheckParamsPassed",
                    "[BayesianLinearRegressionMainTest][BindingTests]")
   {
     int n = 10, m = 4;
     arma::mat matX = arma::randu<arma::mat>(m, n);
     arma::mat matXtest = arma::randu<arma::mat>(m, 2 * n);
     const arma::rowvec omega = arma::randu<arma::rowvec>(m);
     arma::rowvec y =  omega * matX;
   
     BayesianLinearRegression model;
     model.Train(matX, y);
   
     arma::rowvec responses;
     model.Predict(matXtest, responses);
   
     // Check that std::runtime_error is thrown if neither input or input_model
     // is specified.
     SetInputParam("responses", std::move(y));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   
     // Continue only with input passed.
     SetInputParam("input", std::move(matX));
     mlpackMain();
   
     // Now pass the previous trained model and one input matrix at the same time.
     // An error should occur.
     SetInputParam("input", std::move(matX));
     SetInputParam("input_model",
                   IO::GetParam<BayesianLinearRegression*>("output_model"));
     SetInputParam("test", std::move(matXtest));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
