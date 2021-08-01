
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_perceptron_test.cpp:

Program Listing for File perceptron_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_perceptron_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/perceptron_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #define BINDING_TYPE BINDING_TYPE_TEST
   
   #include <mlpack/core.hpp>
   static const std::string testName = "Perceptron";
   
   #include <mlpack/core/util/mlpack_main.hpp>
   #include <mlpack/methods/perceptron/perceptron_main.cpp>
   #include "test_helper.hpp"
   
   #include "../catch.hpp"
   #include "../test_catch_tools.hpp"
   
   using namespace mlpack;
   
   struct PerceptronTestFixture
   {
    public:
     PerceptronTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~PerceptronTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronOutputDimensionTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     arma::mat inputData;
     if (!data::Load("trainSet.csv", inputData))
       FAIL("Cannot load train dataset trainSet.csv!");
   
     // Get the labels out.
     arma::Row<size_t> labels(inputData.n_cols);
     for (size_t i = 0; i < inputData.n_cols; ++i)
       labels[i] = inputData(inputData.n_rows - 1, i);
   
     // Delete the last row containing labels from input dataset.
     inputData.shed_row(inputData.n_rows - 1);
   
     arma::mat testData;
     if (!data::Load("testSet.csv", testData))
       FAIL("Cannot load test dataset testSet.csv!");
   
     // Delete the last row containing labels from test dataset.
     testData.shed_row(testData.n_rows - 1);
   
     size_t testSize = testData.n_cols;
   
     // Input training data.
     SetInputParam("training", std::move(inputData));
     SetInputParam("labels", std::move(labels));
   
     // Input test data.
     SetInputParam("test", std::move(testData));
   
     mlpackMain();
   
     // Check that number of output points are equal to number of input points.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
   
     // Check output have only single row.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronLabelsLessDimensionTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     // Train perceptron without providing labels.
     arma::mat inputData;
     if (!data::Load("trainSet.csv", inputData))
       FAIL("Cannot load train dataset trainSet.csv!");
   
     // Get the labels out.
     arma::Row<size_t> labels(inputData.n_cols);
     for (size_t i = 0; i < inputData.n_cols; ++i)
       labels[i] = inputData(inputData.n_rows - 1, i);
   
     arma::mat testData;
     if (!data::Load("testSet.csv", testData))
       FAIL("Cannot load test dataset testSet.csv!");
   
     // Delete the last row containing labels from test dataset.
     testData.shed_row(testData.n_rows - 1);
   
     size_t testSize = testData.n_cols;
   
     // Input training data.
     SetInputParam("training", inputData);
   
     // Input test data.
     SetInputParam("test", testData);
   
     mlpackMain();
   
     // Check that number of output points are equal to number of input points.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
   
     // Check output have only single row.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
   
     // Reset data passed.
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["test"].wasPassed = false;
   
     inputData.shed_row(inputData.n_rows - 1);
   
     // Store outputs.
     arma::Row<size_t> output;
     output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
   
     bindings::tests::CleanMemory();
   
     // Now train perceptron with labels provided.
   
     // Input training data.
     SetInputParam("training", std::move(inputData));
     SetInputParam("test", std::move(testData));
     // Pass Labels.
     SetInputParam("labels", std::move(labels));
   
     mlpackMain();
   
     // Check that number of output points are equal to number of input points.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
   
     // Check output have only single row.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
   
     // Check that initial output and final output matrix
     // from two models are same.
     CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronOutputPredictionsCheck",
                    "[PerceptronMainTest][BindingTests]")
   {
     arma::mat trainX1;
     arma::Row<size_t> labelsX1;
   
     // Loading a train data set with 3 classes.
     if (!data::Load("vc2.csv", trainX1))
     {
       FAIL("Could not load the train data (vc2.csv)");
     }
   
     // Loading the corresponding labels to the dataset.
     if (!data::Load("vc2_labels.txt", labelsX1))
     {
       FAIL("Could not load the train data (vc2_labels.csv)");
     }
   
     SetInputParam("training", std::move(trainX1)); // Training data.
     // Labels for the training data.
     SetInputParam("labels", std::move(labelsX1));
   
     // Training model using first training dataset.
     mlpackMain();
   
     // Check that the outputs are the same.
     CheckMatrices(IO::GetParam<arma::Row<size_t>>("output"),
                   IO::GetParam<arma::Row<size_t>>("predictions"));
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronModelReuseTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     arma::mat inputData;
     if (!data::Load("trainSet.csv", inputData))
       FAIL("Cannot load train dataset trainSet.csv!");
   
     arma::mat testData;
     if (!data::Load("testSet.csv", testData))
       FAIL("Cannot load test dataset testSet.csv!");
   
     // Delete the last row containing labels from test dataset.
     testData.shed_row(testData.n_rows - 1);
   
     size_t testSize = testData.n_cols;
   
     // Input training data.
     SetInputParam("training", std::move(inputData));
   
     // Input test data.
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::Row<size_t> output;
     output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
   
     // Reset passed parameters.
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["test"].wasPassed = false;
   
     // Input trained model.
     SetInputParam("test", std::move(testData));
     SetInputParam("input_model",
                   IO::GetParam<PerceptronModel*>("output_model"));
   
     mlpackMain();
   
     // Check that number of output points are equal to number of input points.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
   
     // Check output have only single row.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
   
     // Check that initial output and final output matrix
     // using saved model are same.
     CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronMaxItrTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     arma::mat inputData;
     if (!data::Load("trainSet.csv", inputData))
       FAIL("Cannot load train dataset trainSet.csv!");
   
     // Input training data.
     SetInputParam("training", std::move(inputData));
     SetInputParam("max_iterations", (int) -1);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronReTrainWithWrongClasses",
                    "[PerceptronMainTest][BindingTests]")
   {
     arma::mat trainX1;
     arma::Row<size_t> labelsX1;
   
     // Loading a train data set with 3 classes.
     if (!data::Load("vc2.csv", trainX1))
     {
       FAIL("Could not load the train data (vc2.csv)");
     }
   
     // Loading the corresponding labels to the dataset.
     if (!data::Load("vc2_labels.txt", labelsX1))
     {
       FAIL("Could not load the train data (vc2_labels.csv)");
     }
   
     SetInputParam("training", std::move(trainX1)); // Training data.
     // Labels for the training data.
     SetInputParam("labels", std::move(labelsX1));
   
     // Training model using first training dataset.
     mlpackMain();
   
     // Get the output model obtained after training.
     PerceptronModel* model =
         IO::GetParam<PerceptronModel*>("output_model");
   
     // Reset the data passed.
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["labels"].wasPassed = false;
   
     // Creating training data with five classes.
     constexpr int D = 3;
     constexpr int N = 10;
     arma::mat trainX2 = arma::randu<arma::mat>(D, N);
     arma::Row<size_t> labelsX2;
   
     // 10 responses.
     labelsX2 = { 0, 1, 4, 1, 2, 1, 0, 3, 3, 0 };
   
     // Last column of trainX2 contains the class labels.
     SetInputParam("training", std::move(trainX2));
     SetInputParam("input_model", model);
   
     // Re-training an existing model of 3 classes
     // with training data of 5 classes. It should give runtime error.
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronWrongDimOfTestData",
                    "[PerceptronMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 4;
     constexpr int M = 20;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::Row<size_t> trainY;
   
     // 10 responses.
     trainY = { 0 , 1, 0, 1, 1, 1, 0, 1, 0, 0 };
   
     // Test data with wrong dimensionality.
     arma::mat testX = arma::randu<arma::mat>(D-3, M);
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("labels", std::move(trainY));
     SetInputParam("test", std::move(testX));
   
     // Test data set with wrong dimensionality. It should give runtime error.
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronWrongResponseSizeTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     constexpr int D = 2;
     constexpr int N = 10;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::Row<size_t> trainY; // Response vector with wrong size.
   
     // 8 responses.
     trainY = { 0, 0, 1, 0, 1, 1, 1, 0 };
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("labels", std::move(trainY));
   
     // Labels for training data have wrong size. It should give runtime error.
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronNoResponsesTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 1;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     SetInputParam("training", std::move(trainX));
   
     // No labels for training data. It should give runtime error.
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronNoTrainingDataTest",
                    "[PerceptronMainTest][BindingTests]")
   {
     arma::Row<size_t> trainY;
     trainY = { 1, 1, 0, 1, 0, 0 };
   
     SetInputParam("labels", std::move(trainY));
   
     // No training data. It should give runtime error.
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(PerceptronTestFixture, "PerceptronWrongDimOfTestData2",
                    "[PerceptronMainTest][BindingTests]")
   {
     constexpr int N = 10;
     constexpr int D = 3;
     constexpr int M = 15;
   
     arma::mat trainX = arma::randu<arma::mat>(D, N);
     arma::Row<size_t> trainY;
   
     // 10 responses.
     trainY = { 0, 1, 0, 1, 1, 1, 0, 1, 0, 0 };
   
     SetInputParam("training", std::move(trainX));
     SetInputParam("labels", std::move(trainY));
   
     // Training the model.
     mlpackMain();
   
     // Get the output model obtained after the training.
     PerceptronModel* model =
         IO::GetParam<PerceptronModel*>("output_model");
   
     // Reset the data passed.
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["labels"].wasPassed = false;
   
     // Test data with Wrong dimensionality.
     arma::mat testX = arma::randu<arma::mat>(D - 1, M);
     SetInputParam("input_model", model);
     SetInputParam("test", std::move(testX));
   
     // Wrong dimensionality of test data. It should give runtime error.
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
