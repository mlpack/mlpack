
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_main_tests_adaboost_test.cpp:

Program Listing for File adaboost_test.cpp
==========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_main_tests_adaboost_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/main_tests/adaboost_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <string>
   
   #define BINDING_TYPE BINDING_TYPE_TEST
   static const std::string testName = "AdaBoost";
   
   #include <mlpack/core.hpp>
   #include <mlpack/core/util/mlpack_main.hpp>
   #include "test_helper.hpp"
   #include <mlpack/methods/adaboost/adaboost_main.cpp>
   
   #include "../test_catch_tools.hpp"
   #include "../catch.hpp"
   
   using namespace mlpack;
   
   struct AdaBoostTestFixture
   {
    public:
     AdaBoostTestFixture()
     {
       // Cache in the options for this program.
       IO::RestoreSettings(testName);
     }
   
     ~AdaBoostTestFixture()
     {
       // Clear the settings.
       bindings::tests::CleanMemory();
       IO::ClearSettings();
     }
   };
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostOutputDimensionTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     arma::mat testData;
     if (!data::Load("vc2_test.csv", testData))
       FAIL("Unable to load test dataset vc2.csv!");
   
     size_t testSize = testData.n_cols;
   
     SetInputParam("training", std::move(trainData));
     SetInputParam("labels", std::move(labels));
   
     SetInputParam("test", std::move(testData));
   
     mlpackMain();
   
     // Check that number of predicted labels is equal to the input test points.
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_cols == testSize);
     REQUIRE(IO::GetParam<arma::Row<size_t>>("output").n_rows == 1);
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostProbabilitiesTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     arma::mat testData;
     if (!data::Load("vc2_test.csv", testData))
       FAIL("Unable to load test dataset vc2.csv!");
   
     size_t testSize = testData.n_cols;
   
     SetInputParam("training", std::move(trainData));
     SetInputParam("labels", std::move(labels));
   
     SetInputParam("test", std::move(testData));
   
     mlpackMain();
   
     arma::mat probabilities;
     probabilities = std::move(IO::GetParam<arma::mat>("probabilities"));
   
     REQUIRE(probabilities.n_cols == testSize);
   
     for (size_t i = 0; i < testSize; ++i)
       REQUIRE(arma::accu(probabilities.col(i)) == Approx(1).epsilon(1e-7));
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostModelReuseTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     arma::mat testData;
     if (!data::Load("vc2_test.csv", testData))
       FAIL("Unable to load test dataset vc2.csv!");
   
     SetInputParam("training", std::move(trainData));
     SetInputParam("labels", std::move(labels));
   
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::Row<size_t> output;
     output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
   
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["labels"].wasPassed = false;
     IO::GetSingleton().Parameters()["test"].wasPassed = false;
   
     SetInputParam("test", std::move(testData));
     SetInputParam("input_model",
                   IO::GetParam<AdaBoostModel*>("output_model"));
   
     mlpackMain();
   
     // Check that initial output and output using saved model are same.
     CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostItrTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("trainSet.csv", trainData))
       FAIL("Unable load train dataset trainSet.csv!");
   
     SetInputParam("training", std::move(trainData));
     SetInputParam("iterations", (int) -1);
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostWithoutLabelTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     // Train adaboost without providing labels.
     arma::mat trainData;
     if (!data::Load("trainSet.csv", trainData))
       FAIL("Unable to load train dataset trainSet.csv!");
   
     // Give labels.
     arma::Row<size_t> labels(trainData.n_cols);
     for (size_t i = 0; i < trainData.n_cols; ++i)
       labels[i] = trainData(trainData.n_rows - 1, i);
   
     arma::mat testData;
     if (!data::Load("testSet.csv", testData))
       FAIL("Unable to load test dataset testSet.csv!");
   
     // Delete the last row containing labels from test dataset.
     testData.shed_row(testData.n_rows - 1);
   
     SetInputParam("training", trainData);
   
     SetInputParam("test", testData);
   
     mlpackMain();
   
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["test"].wasPassed = false;
   
     arma::Row<size_t> output;
     output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
   
     bindings::tests::CleanMemory();
   
     trainData.shed_row(trainData.n_rows - 1);
   
     // Now train Adaboost with labels provided.
     SetInputParam("training", std::move(trainData));
     SetInputParam("test", std::move(testData));
     SetInputParam("labels", std::move(labels));
   
     mlpackMain();
   
     // Check that initial output and final output matrix are same.
     CheckMatrices(output, IO::GetParam<arma::Row<size_t>>("output"));
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostTrainingDataOrModelTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("trainSet.csv", trainData))
       FAIL("Unable to load train dataset trainSet.csv!");
   
     SetInputParam("training", std::move(trainData));
   
     mlpackMain();
   
     SetInputParam("input_model",
                   IO::GetParam<AdaBoostModel*>("output_model"));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostOutputPredictionsTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     SetInputParam("training", std::move(trainData));
     SetInputParam("labels", std::move(labels));
   
     mlpackMain();
   
     CheckMatrices(IO::GetParam<arma::Row<size_t>>("output"),
                   IO::GetParam<arma::Row<size_t>>("predictions"));
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostWeakLearnerTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("trainSet.csv", trainData))
       FAIL("Unable to load train dataset trainSet.csv!");
   
     SetInputParam("training", std::move(trainData));
     SetInputParam("weak_learner", std::string("decision tree"));
   
     Log::Fatal.ignoreInput = true;
     REQUIRE_THROWS_AS(mlpackMain(), std::runtime_error);
     Log::Fatal.ignoreInput = false;
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostDiffWeakLearnerOutputTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     arma::mat testData;
     if (!data::Load("vc2_test.csv", testData))
       FAIL("Unable to load test dataset vc2.csv!");
   
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("test", testData);
   
     mlpackMain();
   
     arma::Row<size_t> output;
     output = std::move(IO::GetParam<arma::Row<size_t>>("output"));
   
     bindings::tests::CleanMemory();
   
     IO::GetSingleton().Parameters()["training"].wasPassed = false;
     IO::GetSingleton().Parameters()["labels"].wasPassed = false;
     IO::GetSingleton().Parameters()["test"].wasPassed = false;
   
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("test", testData);
     SetInputParam("weak_learner", std::string("perceptron"));
   
     mlpackMain();
   
     arma::Row<size_t> outputPerceptron;
     outputPerceptron = std::move(IO::GetParam<arma::Row<size_t>>("output"));
   
     REQUIRE(arma::accu(output != outputPerceptron) > 1);
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostDiffItrTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     arma::mat testData;
     if (!data::Load("vc2_test.csv", testData))
       FAIL("Unable to load test dataset vc2.csv!");
   
     arma::Row<size_t> testLabels;
     if (!data::Load("vc2_test_labels.txt", testLabels))
       FAIL("Unable to load labels for vc2__test_labels.txt");
   
     // Iterations = 1
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("weak_learner", std::string("perceptron"));
     SetInputParam("iterations", (int) 1);
   
     mlpackMain();
   
     // Calculate accuracy.
     arma::Row<size_t> output;
     IO::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
          output);
   
     size_t correct = arma::accu(output == testLabels);
     double accuracy1 = (double(correct) / double(testLabels.n_elem) * 100);
   
     bindings::tests::CleanMemory();
   
     // Iterations = 10
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("weak_learner", std::string("perceptron"));
     SetInputParam("iterations", (int) 10);
   
     mlpackMain();
   
     // Calculate accuracy.
     IO::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
          output);
   
     correct = arma::accu(output == testLabels);
     double accuracy10 = (double(correct) / double(testLabels.n_elem) * 100);
   
     bindings::tests::CleanMemory();
   
     // Iterations = 100
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("weak_learner", std::string("perceptron"));
     SetInputParam("iterations", (int) 100);
   
     mlpackMain();
   
     // Calculate accuracy.
     IO::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
          output);
   
     correct = arma::accu(output == testLabels);
     double accuracy100 = (double(correct) / double(testLabels.n_elem) * 100);
   
     REQUIRE(accuracy1 <= accuracy10);
     REQUIRE(accuracy10 <= accuracy100);
   }
   
   TEST_CASE_METHOD(AdaBoostTestFixture, "AdaBoostDiffTolTest",
                    "[AdaBoostMainTest][BindingTests]")
   {
     arma::mat trainData;
     if (!data::Load("vc2.csv", trainData))
       FAIL("Unable to load train dataset vc2.csv!");
   
     arma::Row<size_t> labels;
     if (!data::Load("vc2_labels.txt", labels))
       FAIL("Unable to load label dataset vc2_labels.txt!");
   
     arma::mat testData;
     if (!data::Load("vc2_test.csv", testData))
       FAIL("Unable to load test dataset vc2.csv!");
   
     arma::Row<size_t> testLabels;
     if (!data::Load("vc2_test_labels.txt", testLabels))
       FAIL("Unable to load labels for vc2__test_labels.txt");
   
     // tolerance = 0.001
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("tolerance", (double) 0.001);
   
     mlpackMain();
   
     // Calculate accuracy.
     arma::Row<size_t> output;
     IO::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
          output);
   
     size_t correct = arma::accu(output == testLabels);
     double accuracy1 = (double(correct) / double(testLabels.n_elem) * 100);
   
     bindings::tests::CleanMemory();
   
     // tolerance = 0.01
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("tolerance", (double) 0.01);
   
     mlpackMain();
   
     // Calculate accuracy.
     IO::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
          output);
   
     correct = arma::accu(output == testLabels);
     double accuracy2 = (double(correct) / double(testLabels.n_elem) * 100);
   
     bindings::tests::CleanMemory();
   
     // tolerance = 0.1
     SetInputParam("training", trainData);
     SetInputParam("labels", labels);
     SetInputParam("tolerance", (double) 0.1);
   
     mlpackMain();
   
     // Calculate accuracy.
     IO::GetParam<AdaBoostModel*>("output_model")->Classify(testData,
          output);
   
     correct = arma::accu(output == testLabels);
     double accuracy3 = (double(correct) / double(testLabels.n_elem) * 100);
   
     REQUIRE(accuracy1 <= accuracy2);
     REQUIRE(accuracy2 <= accuracy3);
   }
