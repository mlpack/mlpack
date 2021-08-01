
.. _program_listing_file__home_aakash_mlpack_src_mlpack_tests_ann_visitor_test.cpp:

Program Listing for File ann_visitor_test.cpp
=============================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_mlpack_src_mlpack_tests_ann_visitor_test.cpp>` (``/home/aakash/mlpack/src/mlpack/tests/ann_visitor_test.cpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/visitor/bias_set_visitor.hpp>
   #include <mlpack/methods/ann/visitor/weight_set_visitor.hpp>
   #include <mlpack/methods/ann/visitor/reset_visitor.hpp>
   
   #include "catch.hpp"
   #include "test_catch_tools.hpp"
   
   using namespace mlpack;
   using namespace mlpack::ann;
   
   TEST_CASE("BiasSetVisitorTest", "[ANNVisitorTest]")
   {
     LayerTypes<> linear = new Linear<>(10, 10);
   
     arma::mat layerWeights(110, 1);
     layerWeights.zeros();
   
     ResetVisitor resetVisitor;
   
     boost::apply_visitor(WeightSetVisitor(layerWeights, 0), linear);
   
     boost::apply_visitor(resetVisitor, linear);
   
     arma::mat weight = {"1 2 3 4 5 6 7 8 9 10"};
   
     size_t biasSize = boost::apply_visitor(BiasSetVisitor(weight, 0), linear);
   
     REQUIRE(biasSize == 10);
   
     arma::mat input(10, 1), output;
     input.randu();
   
     boost::apply_visitor(ForwardVisitor(input, output), linear);
   
     REQUIRE(arma::accu(output) == 55);
   
     boost::apply_visitor(DeleteVisitor(), linear);
   }
   
   void CheckCorrectnessOfWeightSize(LayerTypes<>& layer)
   {
     size_t weightSize = boost::apply_visitor(WeightSizeVisitor(),
         layer);
   
     arma::mat parameters;
     boost::apply_visitor(ParametersVisitor(parameters), layer);
   
     REQUIRE(weightSize == parameters.n_elem);
   }
   
   TEST_CASE("WeightSetVisitorTest", "[ANNVisitorTest]")
   {
     size_t randomSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> linear = new Linear<>(randomSize, randomSize);
   
     arma::mat layerWeights(randomSize * randomSize + randomSize, 1);
     layerWeights.zeros();
   
     size_t setWeights = boost::apply_visitor(WeightSetVisitor(layerWeights, 0),
         linear);
   
     REQUIRE(setWeights == randomSize * randomSize + randomSize);
   }
   
   TEST_CASE("WeightSizeVisitorTestForLinearLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> linearLayer = new Linear<>(randomInSize, randomOutSize);
   
     CheckCorrectnessOfWeightSize(linearLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForConcatLayer", "[ANNVisitorTest]")
   {
     LayerTypes<> concatLayer = new Concat<>();
   
     CheckCorrectnessOfWeightSize(concatLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForFastLSTMLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> fastLSTMLayer = new FastLSTM<>(randomInSize, randomOutSize);
   
     CheckCorrectnessOfWeightSize(fastLSTMLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForAddLayer", "[ANNVisitorTest]")
   {
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> addLayer = new Add<>(randomOutSize);
   
     CheckCorrectnessOfWeightSize(addLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForAtrousConvolutionLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
     size_t randomKernelWidth = arma::randi(arma::distr_param(1, 100));
     size_t randomKernelHeight = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> atrousConvLayer = new AtrousConvolution<>(randomInSize,
         randomOutSize, randomKernelWidth, randomKernelHeight);
   
     CheckCorrectnessOfWeightSize(atrousConvLayer);
   }
   
   
   TEST_CASE("WeightSizeVisitorTestForConvLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
     size_t randomKernelWidth = arma::randi(arma::distr_param(1, 100));
     size_t randomKernelHeight = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> convLayer = new Convolution<>(randomInSize, randomOutSize,
         randomKernelWidth, randomKernelHeight);
     CheckCorrectnessOfWeightSize(convLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForBatchNormLayer", "[ANNVisitorTest]")
   {
     size_t randomSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> batchNorm = new BatchNorm<>(randomSize);
     CheckCorrectnessOfWeightSize(batchNorm);
   }
   
   TEST_CASE("WeightSizeVisitorTestForLSTMLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> lstm = new LSTM<>(randomInSize, randomOutSize);
     CheckCorrectnessOfWeightSize(lstm);
   }
   
   TEST_CASE("WeightSizeVisitorTestForTransposedConvLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
     size_t randomKernelWidth = arma::randi(arma::distr_param(1, 100));
     size_t randomKernelHeight = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> transposedConvLayer = new TransposedConvolution<>(randomInSize,
         randomOutSize, randomKernelWidth, randomKernelHeight);
   
     CheckCorrectnessOfWeightSize(transposedConvLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForNoisyLinearLayer", "[ANNVisitorTest]")
   {
     size_t randomInSize = arma::randi(arma::distr_param(1, 100));
     size_t randomOutSize = arma::randi(arma::distr_param(1, 100));
   
     LayerTypes<> noisyLinearLayer = new NoisyLinear<>(randomInSize,
         randomOutSize);
   
     CheckCorrectnessOfWeightSize(noisyLinearLayer);
   }
   
   TEST_CASE("WeightSizeVisitorTestForMultiheadAttentionLayer", "[ANNVisitorTest]")
   {
     size_t randomtgtSeqLen = arma::randi(arma::distr_param(1, 100));
     size_t randomsrcSeqLen = arma::randi(arma::distr_param(1, 100));
     size_t randomembedDim = 768;
     size_t randomnumHeads = 12;
   
     LayerTypes<> MultiheadAttentionLayer = new MultiheadAttention<>(
         randomtgtSeqLen, randomsrcSeqLen, randomembedDim, randomnumHeads);
   
     CheckCorrectnessOfWeightSize(MultiheadAttentionLayer);
   }
