
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_perceptron_test.cpp:

Program Listing for File perceptron_test.cpp
============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_perceptron_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/perceptron_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/perceptron/perceptron.hpp>
   #include <mlpack/methods/perceptron/learning_policies/simple_weight_update.hpp>
   
   #include "catch.hpp"
   
   using namespace mlpack;
   using namespace arma;
   using namespace mlpack::perceptron;
   using namespace mlpack::distribution;
   
   TEST_CASE("SimpleWeightUpdateWeights", "[PerceptronTest]")
   {
     SimpleWeightUpdate wip;
   
     vec trainingPoint("1 2 3 4 5");
     mat weights("0 1 6;"
                 "2 3 6;"
                 "4 5 6;"
                 "6 7 6;"
                 "8 9 6");
     vec biases("2 5 7");
     size_t incorrectClass = 0;
     size_t correctClass = 2;
   
     wip.UpdateWeights(trainingPoint, weights, biases, incorrectClass,
                       correctClass);
   
     CHECK(weights(0, 0) == -1);
     CHECK(weights(1, 0) == 0);
     CHECK(weights(2, 0) == 1);
     CHECK(weights(3, 0) == 2);
     CHECK(weights(4, 0) == 3);
   
     CHECK(weights(0, 2) == 7);
     CHECK(weights(1, 2) == 8);
     CHECK(weights(2, 2) == 9);
     CHECK(weights(3, 2) == 10);
     CHECK(weights(4, 2) == 11);
   
     CHECK(biases(0) == 1);
     CHECK(biases(2) == 8);
   }
   
   TEST_CASE("SimpleWeightUpdateInstanceWeight", "[PerceptronTest]")
   {
     SimpleWeightUpdate wip;
   
     vec trainingPoint("1 2 3 4 5");
     mat weights("0 1 6;"
                 "2 3 6;"
                 "4 5 6;"
                 "6 7 6;"
                 "8 9 6");
     vec biases("2 5 7");
     size_t incorrectClass = 0;
     size_t correctClass = 2;
     double instanceWeight = 3.0;
   
     wip.UpdateWeights(trainingPoint, weights, biases, incorrectClass,
                       correctClass, instanceWeight);
   
     CHECK(weights(0, 0) == -3);
     CHECK(weights(1, 0) == -4);
     CHECK(weights(2, 0) == -5);
     CHECK(weights(3, 0) == -6);
     CHECK(weights(4, 0) == -7);
   
     CHECK(weights(0, 2) == 9);
     CHECK(weights(1, 2) == 12);
     CHECK(weights(2, 2) == 15);
     CHECK(weights(3, 2) == 18);
     CHECK(weights(4, 2) == 21);
   
     CHECK(biases(0) == -1);
     CHECK(biases(2) == 10);
   }
   
   TEST_CASE("And", "[PerceptronTest]")
   {
     mat trainData;
     trainData = { { 0, 1, 1, 0 },
                   { 1, 0, 1, 0 } };
     Mat<size_t> labels;
     labels = { 0, 0, 1, 0 };
   
     Perceptron<> p(trainData, labels.row(0), 2, 1000);
   
     mat testData;
     testData = { { 0, 1, 1, 0 },
                  { 1, 0, 1, 0 } };
     Row<size_t> predictedLabels(testData.n_cols);
     p.Classify(testData, predictedLabels);
   
     CHECK(predictedLabels(0, 0) == 0);
     CHECK(predictedLabels(0, 1) == 0);
     CHECK(predictedLabels(0, 2) == 1);
     CHECK(predictedLabels(0, 3) == 0);
   }
   
   TEST_CASE("Or", "[PerceptronTest]")
   {
     mat trainData;
     trainData = { { 0, 1, 1, 0 },
                   { 1, 0, 1, 0 } };
   
     Mat<size_t> labels;
     labels = { 1, 1, 1, 0 };
   
     Perceptron<> p(trainData, labels.row(0), 2, 1000);
   
     mat testData;
     testData = { { 0, 1, 1, 0 },
                  { 1, 0, 1, 0 } };
     Row<size_t> predictedLabels(testData.n_cols);
     p.Classify(testData, predictedLabels);
   
     CHECK(predictedLabels(0, 0) == 1);
     CHECK(predictedLabels(0, 1) == 1);
     CHECK(predictedLabels(0, 2) == 1);
     CHECK(predictedLabels(0, 3) == 0);
   }
   
   TEST_CASE("Random3", "[PerceptronTest]")
   {
     mat trainData;
     trainData = { { 0, 1, 1, 4, 5, 4, 1, 2, 1 },
                   { 1, 0, 1, 1, 1, 2, 4, 5, 4 } };
   
     Mat<size_t> labels;
     labels = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
   
     Perceptron<> p(trainData, labels.row(0), 3, 1000);
   
     mat testData;
     testData = { { 0, 1, 1 },
                  { 1, 0, 1 } };
     Row<size_t> predictedLabels(testData.n_cols);
     p.Classify(testData, predictedLabels);
   
     for (size_t i = 0; i < predictedLabels.n_cols; ++i)
       CHECK(predictedLabels(0, i) == 0);
   }
   
   TEST_CASE("TwoPoints", "[PerceptronTest]")
   {
     mat trainData;
     trainData = { { 0, 1 },
                   { 1, 0 } };
   
     Mat<size_t> labels;
     labels = { 0, 1 };
   
     Perceptron<> p(trainData, labels.row(0), 2, 1000);
   
     mat testData;
     testData = { { 0, 1 },
                  { 1, 0 } };
     Row<size_t> predictedLabels(testData.n_cols);
     p.Classify(testData, predictedLabels);
   
     CHECK(predictedLabels(0, 0) == 0);
     CHECK(predictedLabels(0, 1) == 1);
   }
   
   TEST_CASE("NonLinearlySeparableDataset", "[PerceptronTest]")
   {
     mat trainData;
     trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                   { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };
   
     Mat<size_t> labels;
     labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };
   
     Perceptron<> p(trainData, labels.row(0), 2, 1000);
   
     mat testData;
     testData = { { 3,   4,   5,   6 },
                  { 3, 2.3, 1.7, 1.5 } };
     Row<size_t> predictedLabels(testData.n_cols);
     p.Classify(testData, predictedLabels);
   
     CHECK(predictedLabels(0, 0) == 0);
     CHECK(predictedLabels(0, 1) == 0);
     CHECK(predictedLabels(0, 2) == 1);
     CHECK(predictedLabels(0, 3) == 1);
   }
   
   TEST_CASE("SecondaryConstructor", "[PerceptronTest]")
   {
     mat trainData;
     trainData = { { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 },
                   { 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 } };
   
     Mat<size_t> labels;
     labels = { 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1 };
   
     Perceptron<> p1(trainData, labels.row(0), 2, 1000);
   
     Perceptron<> p2(p1);
   }
