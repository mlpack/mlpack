
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_linear_regression_test.cpp:

Program Listing for File linear_regression_test.cpp
===================================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_linear_regression_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/linear_regression_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "LinearRegression";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/linear_regression/linear_regression_main.cpp>
   
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct LRTestFixture
   {
    public:
     LRTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~LRTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   void ResetSettings()
   {
     IO::ClearSettings();
     IO::RestoreSettings(testName);
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRDifferentLambdas",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     // A required minimal difference between solutions.
     const double delta = 0.1;
   
     arma::mat trainX({1.0, 2.0, 3.0});
     arma::mat testX({4.0});
     arma::rowvec trainY({1.0, 4.0, 9.0});
   
     SetInputParam("training", trainX);
     SetInputParam("training_responses", trainY);
     SetInputParam("test", testX);
     SetInputParam("lambda", 0.1);
   
     // The first solution.
     mlpackMain();
     const double testY1 = IO::GetParam<arma::rowvec>("output_predictions")(0);
   
     bindings::tests::CleanMemory();
     ResetSettings();
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("training_responses", std::move(trainY));
     SetInputParam("test", std::move(testX));
     SetInputParam("lambda", 1.0);
   
     // The second solution.
     mlpackMain();
     const double testY2 = IO::GetParam<arma::rowvec>("output_predictions")(0);
   
     // Second solution has stronger regularization,
     // so the predicted value should be smaller.
     REQUIRE(testY1 - delta > testY2);
   }
   
   
   TEST_CASE_METHOD(LRTestFixture, "LRResponsesRepresentation",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr double delta = 1e-5;
   
     arma::mat trainX1({{1.0, 2.0, 3.0}, {1.0, 4.0, 9.0}});
     arma::mat testX({4.0});
     SetInputParam("training", trainX1);
     SetInputParam("test", testX);
   
     // The first solution.
     mlpackMain();
     const double testY1 = IO::GetParam<arma::rowvec>("output_predictions")(0);
   
     bindings::tests::CleanMemory();
     ResetSettings();
   
     arma::mat trainX2({1.0, 2.0, 3.0});
     arma::rowvec trainY2({1.0, 4.0, 9.0});
     SetInputParam("training", std::move(trainX2));
     SetInputParam("training_responses", std::move(trainY2));
     SetInputParam("test", std::move(testX));
   
     // The second solution.
     mlpackMain();
     const double testY2 = IO::GetParam<arma::rowvec>("output_predictions")(0);
   
     REQUIRE(fabs(testY1 - testY2) < delta);
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRModelReload",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr double delta = 1e-5;
     constexpr int N = 10;
     constexpr int D = 4;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::rowvec trainY = arma::randu<arma::rowvec>(N);
     arma::mat testX = arma::randu<arma::mat>(D, N);
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("training_responses", std::move(trainY));
     SetInputParam("test", testX);
   
     mlpackMain();
   
     LinearRegression* model = IO::GetParam<LinearRegression*>("output_model");
     const arma::rowvec testY1 = IO::GetParam<arma::rowvec>("output_predictions");
   
     ResetSettings();
   
     SetInputParam("input_model", model);
     SetInputParam("test", std::move(testX));
   
     mlpackMain();
   
     const arma::rowvec testY2 = IO::GetParam<arma::rowvec>("output_predictions");
   
     double norm = arma::norm(testY1 - testY2, 2);
     REQUIRE(norm < delta);
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRWrongResponseSizeTest",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 2;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::rowvec trainY = arma::randu<arma::rowvec>(N + 3); // Wrong size.
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("training_responses", std::move(trainY));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRWrongDimOfDataTest1t",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 3;
     constexpr int M = 15;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::rowvec trainY = arma::randu<arma::rowvec>(N);
     arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("training_responses", std::move(trainY));
     SetInputParam("test", std::move(testX));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRWrongDimOfDataTest2",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 3;
     constexpr int M = 15;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::rowvec trainY = arma::randu<arma::rowvec>(N);
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("training_responses", std::move(trainY));
   
     mlpackMain();
   
     LinearRegression* model = IO::GetParam<LinearRegression*>("output_model");
   
     ResetSettings();
   
     arma::mat testX = arma::randu<arma::mat>(D - 1, M); // Wrong dimensionality.
     SetInputParam("input_model", std::move(model));
     SetInputParam("test", std::move(testX));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRPredictionSizeCheck",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 3;
     constexpr int M = 15;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::rowvec trainY = arma::randu<arma::rowvec>(N);
     arma::mat testX = arma::randu<arma::mat>(D, M);
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("training_responses", std::move(trainY));
     SetInputParam("test", std::move(testX));
   
     mlpackMain();
   
     const arma::rowvec testY = IO::GetParam<arma::rowvec>("output_predictions");
   
     REQUIRE(testY.n_rows == 1);
     REQUIRE(testY.n_cols == M);
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRNoResponses",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 1;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     SetInputParam("training", std::move(trainX));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(LRTestFixture, "LRNoTrainingData",
                    "[LinearRegressionMainTest][BindingTests]")
   {
     constexpr int N = 10;
   
     arma::rowvec trainY = arma::randu<arma::rowvec>(N);
     SetInputParam("training_responses", std::move(trainY));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
