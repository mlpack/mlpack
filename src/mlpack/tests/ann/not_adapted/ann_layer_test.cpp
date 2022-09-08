/**
 * @file tests/ann_layer_test.cpp
 * @author Marcus Edel
 * @author Praveen Ch
 *
 * Tests the ann layer modules.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include "../test_catch_tools.hpp"
#include "../catch.hpp"
#include "../serialization.hpp"
#include "ann_test_tools.hpp"

using namespace mlpack;

// // network1 should be allocated with `new`, and trained on some data.
// template<typename MatType = arma::cube, typename ModelType>
// void CheckRNNCopyFunction(ModelType* network1,
//                        MatType& trainData,
//                        MatType& trainLabels,
//                        const size_t maxEpochs)
// {
//   arma::cube predictions1;
//   arma::cube predictions2;
//   ens::StandardSGD opt(0.1, 1, maxEpochs * trainData.n_slices, -100, false);

//   network1->Train(trainData, trainLabels, opt);
//   network1->Predict(trainData, predictions1);

//   RNN<> network2 = *network1;
//   delete network1;

//   // Deallocating all of network1's memory, so that network2 does not use any
//   // of that memory.
//   network2.Predict(trainData, predictions2);
//   CheckMatrices(predictions1, predictions2);
// }

// // network1 should be allocated with `new`, and trained on some data.
// template<typename MatType = arma::cube, typename ModelType>
// void CheckRNNMoveFunction(ModelType* network1,
//                        MatType& trainData,
//                        MatType& trainLabels,
//                        const size_t maxEpochs)
// {
//   arma::cube predictions1;
//   arma::cube predictions2;
//   ens::StandardSGD opt(0.1, 1, maxEpochs * trainData.n_slices, -100, false);

//   network1->Train(trainData, trainLabels, opt);
//   network1->Predict(trainData, predictions1);

//   RNN<> network2(std::move(*network1));
//   delete network1;

//   // Deallocating all of network1's memory, so that network2 does not use any
//   // of that memory.
//   network2.Predict(trainData, predictions2);
//   CheckMatrices(predictions1, predictions2);
// }

/**
 * Simple add module test.
 *
TEST_CASE("SimpleAddLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  Add module(10);
  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  REQUIRE(arma::accu(module.Parameters()) == arma::accu(output));

  // Test the Backward function.
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(output) == arma::accu(delta));

  // Test the forward function.
  input = arma::ones(10, 1);
  module.Forward(input, output);
  REQUIRE(10 + arma::accu(module.Parameters()) ==
      Approx(arma::accu(output)).epsilon(1e-5));

  // Test the backward function.
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(output) == Approx(arma::accu(delta)).epsilon(1e-5));
}
*/

/**
 * Jacobian add module test.
 *
TEST_CASE("JacobianAddLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t elements = RandInt(2, 1000);
    arma::mat input;
    input.set_size(elements, 1);

    Add module(elements);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Add layer numerical gradient test.
 *
TEST_CASE("GradientAddLayerTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<IdentityLayer>();
      model->Add<Linear>(10, 10);
      model->Add<Add>(10);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}*/

/**
 * Test that the function that can access the outSize parameter of
 * the Add layer works.
 *
TEST_CASE("AddLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter : outSize.
  Add layer(7);

  // Make sure we can get the parameter successfully.
  REQUIRE(layer.OutputSize() == 7);
}*/

/**
 * Simple constant module test.
 *
TEST_CASE("SimpleConstantLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  Constant module(10, 3.0);

  // Test the Forward function.
  input = arma::zeros(10, 1);
  module.Forward(input, output);
  REQUIRE(arma::accu(output) == 30.0);

  // Test the Backward function.
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0);

  // Test the forward function.
  input = arma::ones(10, 1);
  module.Forward(input, output);
  REQUIRE(arma::accu(output) == 30.0);

  // Test the backward function.
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0);
}*/

/**
 * Jacobian constant module test.
 *
TEST_CASE("JacobianConstantLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t elements = RandInt(2, 1000);
    arma::mat input;
    input.set_size(elements, 1);

    Constant module(elements, 1.0);

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}*/

/**
 * Test that the function that can access the outSize parameter of the
 * Constant layer works.
 *
TEST_CASE("ConstantLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter : outSize.
  Constant layer(7);

  // Make sure we can get the parameter successfully.
  REQUIRE(layer.OutSize() == 7);
}*/

// /**
//  * Simple linear module test.
//  */
// TEST_CASE("SimpleLinearLayerTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta;
//   Linear<> module(10, 10);
//   module.Parameters().randu();
//   module.Reset();

//   // Test the Forward function.
//   input = arma::zeros(10, 1);
//   module.Forward(input, output);
//   REQUIRE(arma::accu(module.Parameters().submat(100,
//           0, module.Parameters().n_elem - 1, 0)) ==
//           Approx(arma::accu(output)).epsilon(1e-5));

//   // Test the Backward function.
//   module.Backward(input, input, delta);
//   REQUIRE(arma::accu(delta) == 0);
// }

// /**
//  * Jacobian linear module test.
//  */
// TEST_CASE("JacobianLinearLayerTest", "[ANNLayerTest]")
// {
//   for (size_t i = 0; i < 5; ++i)
//   {
//     const size_t inputElements = RandInt(2, 1000);
//     const size_t outputElements = RandInt(2, 1000);

//     arma::mat input;
//     input.set_size(inputElements, 1);

//     Linear<> module(inputElements, outputElements);
//     module.Parameters().randu();

//     double error = JacobianTest(module, input);
//     REQUIRE(error <= 1e-5);
//   }
// }

// /**
//  * Linear layer numerical gradient test.
//  */
// TEST_CASE("GradientLinearLayerTest", "[ANNLayerTest]")
// {
//   // Linear function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(10, 1)),
//         target(arma::mat("1"))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(10, 10);
//       model->Add<Linear<> >(10, 2);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

// /**
//  * Simple noisy linear module test.
//  */
// TEST_CASE("SimpleNoisyLinearLayerTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta;
//   NoisyLinear<> module(10, 10);
//   module.Parameters().randu();
//   module.Reset();

//   // Test the Backward function.
//   module.Backward(input, input, delta);
//   REQUIRE(arma::accu(delta) == 0);
// }

// /**
//  * Jacobian noisy linear module test.
//  */
// TEST_CASE("JacobianNoisyLinearLayerTest", "[ANNLayerTest]")
// {
//   const size_t inputElements = RandInt(2, 1000);
//   const size_t outputElements = RandInt(2, 1000);

//   arma::mat input;
//   input.set_size(inputElements, 1);

//   NoisyLinear<> module(inputElements, outputElements);
//   module.Parameters().randu();

//   double error = JacobianTest(module, input);
//   REQUIRE(error <= 1e-5);
// }

// /**
//  * Noisy Linear layer numerical gradient test.
//  */
// TEST_CASE("GradientNoisyLinearLayerTest", "[ANNLayerTest]")
// {
//   // Noisy linear function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(10, 1)),
//         target(arma::mat("1"))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<NoisyLinear<> >(10, 10);
//       model->Add<NoisyLinear<> >(10, 2);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

// /**
//  * Jacobian negative log likelihood module test.
//  */
// TEST_CASE("JacobianNegativeLogLikelihoodLayerTest", "[ANNLayerTest]")
// {
//   for (size_t i = 0; i < 5; ++i)
//   {
//     NegativeLogLikelihood module;
//     const size_t inputElements = RandInt(5, 100);
//     arma::mat input;
//     RandomInitialization init(0, 1);
//     init.Initialize(input, inputElements, 1);

//     arma::mat target(1, 1);
//     target(0) = RandInt(0, inputElements - 2);

//     double error = JacobianPerformanceTest(module, input, target);
//     REQUIRE(error <= 1e-5);
//   }
// }

/**
 * Jacobian LeakyReLU module test.
 *
TEST_CASE("JacobianLeakyReLULayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    LeakyReLU module;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Jacobian FlexibleReLU module test.
 *
TEST_CASE("JacobianFlexibleReLULayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    FlexibleReLU module;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Flexible ReLU layer numerical gradient test.
 *
TEST_CASE("GradientFlexibleReLULayerTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(2, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, RandomInitialization>(
          NegativeLogLikelihood(), RandomInitialization(0.1, 0.5));

      model->ResetData(input, target);
      model->Add<Linear>(2, 2);
      model->Add<LinearNoBias>(2, 5);
      model->Add<FlexibleReLU>(0.05);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, RandomInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
*/

/**
 * Jacobian MultiplyConstant module test.
 *
TEST_CASE("JacobianMultiplyConstantLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    MultiplyConstant module(3.0);

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Check whether copying and moving network with MultiplyConstant is working or
 * not.
 */
// TEST_CASE("CheckCopyMoveMultiplyConstantTest", "[ANNLayerTest]")
// {
//   arma::mat input(2, 1000);
//   input.randu();
//
//   arma::mat output1;
//   arma::mat output2;
//   arma::mat output3;
//   arma::mat output4;
//
//   MultiplyConstant<> *module1 = new MultiplyConstant<>(3.0);
//   module1->Forward(input, output1);
//
//   MultiplyConstant<> module2 = *module1;
//   delete module1;
//
//   module2.Forward(input, output2);
//   CheckMatrices(output1, output2);
//
//   MultiplyConstant<> *module3 = new MultiplyConstant<>(3.0);
//   module3->Forward(input, output3);
//
//   MultiplyConstant<> module4(std::move(*module3));
//   delete module3;
//
//   module4.Forward(input, output4);
//   CheckMatrices(output3, output4);
// }

/**
 * Jacobian HardTanH module test.
 *
TEST_CASE("JacobianHardTanHLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElements = RandInt(2, 1000);

    arma::mat input;
    input.set_size(inputElements, 1);

    HardTanH module;

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Simple select module test.
 *
TEST_CASE("SimpleSelectLayerTest", "[ANNLayerTest]")
{
  // TODO: this needs to be adapted
  arma::mat outputA, outputB, input, delta;

  input = arma::ones(10, 5);
  for (size_t i = 0; i < input.n_cols; ++i)
  {
    input.col(i) *= i;
  }

  // Test the Forward function.
  Select moduleA(3);
  moduleA.Forward(input, outputA);
  REQUIRE(30 == arma::accu(outputA));

  // Test the Forward function.
  Select moduleB(3, 5);
  moduleB.Forward(input, outputB);
  REQUIRE(15 == arma::accu(outputB));

  // Test the Backward function.
  moduleA.Backward(input, outputA, delta);
  REQUIRE(30 == arma::accu(delta));

  // Test the Backward function.
  moduleB.Backward(input, outputA, delta);
  REQUIRE(15 == arma::accu(delta));
}
*/

/**
 * Test that the functions that can access the parameters of the
 * Select layer work.
 *
TEST_CASE("SelectLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : index, elements.
  Select layer(3, 5);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.Index() == 3);
  REQUIRE(layer.NumElements() == 5);
}
*/

/**
 * Simple join module test.
 *
TEST_CASE("SimpleJoinLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  input = arma::ones(10, 5);

  // Test the Forward function.
  Join module;
  module.Forward(input, output);
  REQUIRE(50 == arma::accu(output));

  bool b = output.n_rows == 1 || output.n_cols == 1;
  REQUIRE(b == true);

  // Test the Backward function.
  module.Backward(input, output, delta);
  REQUIRE(50 == arma::accu(delta));

  b = delta.n_rows == input.n_rows && input.n_cols;
  REQUIRE(b == true);
}
*/

// /**
//  * Simple add merge module test.
//  */
// TEST_CASE("SimpleAddMergeLayerTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta;
//   input = arma::ones(10, 1);

//   for (size_t i = 0; i < 5; ++i)
//   {
//     AddMerge<> module(false, false);
//     const size_t numMergeModules = RandInt(2, 10);
//     for (size_t m = 0; m < numMergeModules; ++m)
//     {
//       IdentityLayer<> identityLayer;
//       identityLayer.Forward(input, identityLayer.OutputParameter());

//       module.Add<IdentityLayer<> >(identityLayer);
//     }

//     // Test the Forward function.
//     module.Forward(input, output);
//     REQUIRE(10 * numMergeModules == arma::accu(output));

//     // Test the Backward function.
//     module.Backward(input, output, delta);
//     REQUIRE(arma::accu(output) == arma::accu(delta));
//   }
// }

// /**
//  * Test the LSTM layer with a user defined rho parameter and without.
//  */
// TEST_CASE("LSTMRrhoTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5;
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::ones(1, 1, 5);
//   RandomInitialization init(0.5, 0.5);

//   // Create model with user defined rho parameter.
//   RNN<NegativeLogLikelihood, RandomInitialization> modelA(
//       rho, false, NegativeLogLikelihood(), init);
//   modelA.Add<IdentityLayer<> >();
//   modelA.Add<Linear<> >(1, 10);

//   // Use LSTM layer with rho.
//   modelA.Add<LSTM<> >(10, 3, rho);
//   modelA.Add<LogSoftMax<> >();

//   // Create model without user defined rho parameter.
//   RNN<NegativeLogLikelihood> modelB(
//       rho, false, NegativeLogLikelihood(), init);
//   modelB.Add<IdentityLayer<> >();
//   modelB.Add<Linear<> >(1, 10);

//   // Use LSTM layer with rho = MAXSIZE.
//   modelB.Add<LSTM<> >(10, 3);
//   modelB.Add<LogSoftMax<> >();

//   ens::StandardSGD opt(0.1, 1, 5, -100, false);
//   modelA.Train(input, target, opt);
//   modelB.Train(input, target, opt);

//   CheckMatrices(modelB.Parameters(), modelA.Parameters());
// }

// /**
//  * LSTM layer numerical gradient test.
//  */
// TEST_CASE("GradientLSTMLayerTest", "[ANNLayerTest]")
// {
//   // LSTM function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(1, 1, 5)),
//         target(arma::ones(1, 1, 5))
//     {
//       const size_t rho = 5;

//       model = new RNN<NegativeLogLikelihood>(rho);
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(1, 10);
//       model->Add<LSTM<> >(10, 3, rho);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     RNN<NegativeLogLikelihood>* model;
//     arma::cube input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

// /**
//  * Test that the functions that can modify and access the parameters of the
//  * LSTM layer work.
//  */
// TEST_CASE("LSTMLayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order : inSize, outSize, rho.
//   LSTM<> layer1(1, 2, 3);
//   LSTM<> layer2(1, 2, 4);

//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InSize() == 1);
//   REQUIRE(layer1.OutSize() == 2);
//   REQUIRE(layer1.Rho() == 3);

//   // Now modify the parameters to match the second layer.
//   layer1.Rho() = 4;

//   // Now ensure all the results are the same.
//   REQUIRE(layer1.InSize() == layer2.InSize());
//   REQUIRE(layer1.OutSize() == layer2.OutSize());
//   REQUIRE(layer1.Rho() == layer2.Rho());
// }

// /**
//  * Test the FastLSTM layer with a user defined rho parameter and without.
//  */
// TEST_CASE("FastLSTMRrhoTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5;
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::ones(1, 1, 5);
//   RandomInitialization init(0.5, 0.5);

//   // Create model with user defined rho parameter.
//   RNN<NegativeLogLikelihood, RandomInitialization> modelA(
//       rho, false, NegativeLogLikelihood(), init);
//   modelA.Add<IdentityLayer<> >();
//   modelA.Add<Linear<> >(1, 10);

//   // Use FastLSTM layer with rho.
//   modelA.Add<FastLSTM<> >(10, 3, rho);
//   modelA.Add<LogSoftMax<> >();

//   // Create model without user defined rho parameter.
//   RNN<NegativeLogLikelihood> modelB(
//       rho, false, NegativeLogLikelihood(), init);
//   modelB.Add<IdentityLayer<> >();
//   modelB.Add<Linear<> >(1, 10);

//   // Use FastLSTM layer with rho = MAXSIZE.
//   modelB.Add<FastLSTM<> >(10, 3);
//   modelB.Add<LogSoftMax<> >();

//   ens::StandardSGD opt(0.1, 1, 5, -100, false);
//   modelA.Train(input, target, opt);
//   modelB.Train(input, target, opt);

//   CheckMatrices(modelB.Parameters(), modelA.Parameters());
// }

// /**
//  * FastLSTM layer numerical gradient test.
//  */
// TEST_CASE("GradientFastLSTMLayerTest", "[ANNLayerTest]")
// {
//   // Fast LSTM function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(1, 1, 5)),
//         target(arma::ones(1, 1, 5))
//     {
//       const size_t rho = 5;

//       model = new RNN<NegativeLogLikelihood>(rho);
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(1, 10);
//       model->Add<FastLSTM<> >(10, 3, rho);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     RNN<NegativeLogLikelihood>* model;
//     arma::cube input, target;
//   } function;

//   // The threshold should be << 0.1 but since the Fast LSTM layer uses an
//   // approximation of the sigmoid function the estimated gradient is not
//   // correct.
//   REQUIRE(CheckGradient(function) <= 0.2);
// }

// /**
//  * Test that the functions that can modify and access the parameters of the
//  * Fast LSTM layer work.
//  */
// TEST_CASE("FastLSTMLayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order : inSize, outSize, rho.
//   FastLSTM<> layer1(1, 2, 3);
//   FastLSTM<> layer2(1, 2, 4);

//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InSize() == 1);
//   REQUIRE(layer1.OutSize() == 2);
//   REQUIRE(layer1.Rho() == 3);

//   // Now modify the parameters to match the second layer.
//   layer1.Rho() = 4;

//   // Now ensure all the results are the same.
//   REQUIRE(layer1.InSize() == layer2.InSize());
//   REQUIRE(layer1.OutSize() == layer2.OutSize());
//   REQUIRE(layer1.Rho() == layer2.Rho());
// }

// /**
//  * Check whether copying and moving network with FastLSTM is working or not.
//  */
// TEST_CASE("CheckCopyMoveFastLSTMTest", "[ANNLayerTest]")
// {
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::ones(1, 1, 5);
//   const size_t rho = 5;

//   RNN<NegativeLogLikelihood> *model1 =
//       new RNN<NegativeLogLikelihood>(rho);
//   model1->ResetData(input, target);
//   model1->Add<IdentityLayer<> >();
//   model1->Add<Linear<> >(1, 10);
//   model1->Add<FastLSTM<> >(10, 3, rho);
//   model1->Add<LogSoftMax<> >();

//   RNN<NegativeLogLikelihood> *model2 =
//      new RNN<NegativeLogLikelihood>(rho);
//   model2->ResetData(input, target);
//   model2->Add<IdentityLayer<> >();
//   model2->Add<Linear<> >(1, 10);
//   model2->Add<FastLSTM<> >(10, 3, rho);
//   model2->Add<LogSoftMax<> >();

//   // Check whether copy constructor is working or not.
//   CheckRNNCopyFunction<>(model1, input, target, 1);

//   // Check whether move constructor is working or not.
//   CheckRNNMoveFunction<>(model2, input, target, 1);
// }

// /**
//  * Check whether copying and moving network with LSTM is working or not.
//  */
// TEST_CASE("CheckCopyMoveLSTMTest", "[ANNLayerTest]")
// {
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::ones(1, 1, 5);
//   const size_t rho = 5;

//   RNN<NegativeLogLikelihood> *model1 =
//       new RNN<NegativeLogLikelihood>(rho);
//   model1->ResetData(input, target);
//   model1->Add<IdentityLayer<> >();
//   model1->Add<Linear<> >(1, 10);
//   model1->Add<LSTM<> >(10, 3, rho);
//   model1->Add<LogSoftMax<> >();

//   RNN<NegativeLogLikelihood> *model2 =
//      new RNN<NegativeLogLikelihood>(rho);
//   model2->ResetData(input, target);
//   model2->Add<IdentityLayer<> >();
//   model2->Add<Linear<> >(1, 10);
//   model2->Add<LSTM<> >(10, 3, rho);
//   model2->Add<LogSoftMax<> >();

//   // Check whether copy constructor is working or not.
//   CheckRNNCopyFunction<>(model1, input, target, 1);

//   // Check whether move constructor is working or not.
//   CheckRNNMoveFunction<>(model2, input, target, 1);
// }

// /**
//  * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
//  * state. Besides output, the overloaded function provides read access to cell
//  * state of the LSTM layer.
//  */
// TEST_CASE("ReadCellStateParamLSTMLayerTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5, inputSize = 3, outputSize = 2;

//   // Provide input of all ones.
//   arma::cube input = arma::ones(inputSize, outputSize, rho);

//   arma::mat inputGate, forgetGate, outputGate, hidden;
//   arma::mat outLstm, cellLstm;

//   // LSTM layer.
//   LSTM<> lstm(inputSize, outputSize, rho);
//   lstm.Reset();
//   lstm.ResetCell(rho);

//   // Initialize the weights to all ones.
//   lstm.Parameters().ones();

//   arma::mat inputWeight = arma::ones(outputSize, inputSize);
//   arma::mat outputWeight = arma::ones(outputSize, outputSize);
//   arma::mat bias = arma::ones(outputSize, input.n_cols);
//   arma::mat cellCalc = arma::zeros(outputSize, input.n_cols);
//   arma::mat outCalc = arma::zeros(outputSize, input.n_cols);

//   for (size_t seqNum = 0; seqNum < rho; ++seqNum)
//   {
//       // Wrap a matrix around our data to avoid a copy.
//       arma::mat stepData(input.slice(seqNum).memptr(),
//           input.n_rows, input.n_cols, false, true);

//       // Apply Forward() on LSTM layer.
//       lstm.Forward(stepData, // Input.
//                    outLstm,  // Output.
//                    cellLstm, // Cell state.
//                    false); // Don't write into the cell state.

//       // Compute the value of cell state and output.
//       // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));

//       // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));

//       // z = tanh(W.dot(x) + W.dot(h) + b).
//       hidden = arma::tanh(inputWeight * stepData +
//                      outputWeight * outCalc + bias);

//       // c = f * c + i * z.
//       cellCalc = forgetGate % cellCalc + inputGate % hidden;

//       // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));

//       // h = o * tanh(c).
//       outCalc = outputGate % arma::tanh(cellCalc);

//       CheckMatrices(outLstm, outCalc, 1e-12);
//       CheckMatrices(cellLstm, cellCalc, 1e-12);
//   }
// }

// /**
//  * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
//  * state. Besides output, the overloaded function provides write access to cell
//  * state of the LSTM layer.
//  */
// TEST_CASE("WriteCellStateParamLSTMLayerTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5, inputSize = 3, outputSize = 2;

//   // Provide input of all ones.
//   arma::cube input = arma::ones(inputSize, outputSize, rho);

//   arma::mat inputGate, forgetGate, outputGate, hidden;
//   arma::mat outLstm, cellLstm;
//   arma::mat cellCalc;

//   // LSTM layer.
//   LSTM<> lstm(inputSize, outputSize, rho);
//   lstm.Reset();
//   lstm.ResetCell(rho);

//   // Initialize the weights to all ones.
//   lstm.Parameters().ones();

//   arma::mat inputWeight = arma::ones(outputSize, inputSize);
//   arma::mat outputWeight = arma::ones(outputSize, outputSize);
//   arma::mat bias = arma::ones(outputSize, input.n_cols);
//   arma::mat outCalc = arma::zeros(outputSize, input.n_cols);

//   for (size_t seqNum = 0; seqNum < rho; ++seqNum)
//   {
//       // Wrap a matrix around our data to avoid a copy.
//       arma::mat stepData(input.slice(seqNum).memptr(),
//           input.n_rows, input.n_cols, false, true);

//       if (cellLstm.is_empty())
//       {
//         // Set the cell state to zeros.
//         cellLstm = arma::zeros(outputSize, input.n_cols);
//         cellCalc = arma::zeros(outputSize, input.n_cols);
//       }
//       else
//       {
//         // Set the cell state to zeros.
//         cellLstm = arma::zeros(cellLstm.n_rows, cellLstm.n_cols);
//         cellCalc = arma::zeros(cellCalc.n_rows, cellCalc.n_cols);
//       }

//       // Apply Forward() on the LSTM layer.
//       lstm.Forward(stepData, // Input.
//                    outLstm,  // Output.
//                    cellLstm, // Cell state.
//                    true);  // Write into cell state.

//       // Compute the value of cell state and output.
//       // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));

//       // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));

//       // z = tanh(W.dot(x) + W.dot(h) + b).
//       hidden = arma::tanh(inputWeight * stepData +
//                      outputWeight * outCalc + bias);

//       // c = f * c + i * z.
//       cellCalc = forgetGate % cellCalc + inputGate % hidden;

//       // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));

//       // h = o * tanh(c).
//       outCalc = outputGate % arma::tanh(cellCalc);

//       CheckMatrices(outLstm, outCalc, 1e-12);
//       CheckMatrices(cellLstm, cellCalc, 1e-12);
//   }

//   // Attempting to write empty matrix into cell state.
//   lstm.Reset();
//   lstm.ResetCell(rho);
//   arma::mat stepData(input.slice(0).memptr(),
//       input.n_rows, input.n_cols, false, true);

//   lstm.Forward(stepData, // Input.
//                outLstm,  // Output.
//                cellLstm, // Cell state.
//                true); // Write into cell state.

//   for (size_t seqNum = 1; seqNum < rho; ++seqNum)
//   {
//     arma::mat empty;
//     // Should throw error.
//     REQUIRE_THROWS_AS(lstm.Forward(stepData, // Input.
//                                    outLstm,  // Output.
//                                    empty, // Cell state.
//                                    true),  // Write into cell state.
//                                    std::runtime_error);
//   }
// }

// /**
//  * Test that the functions that can modify and access the parameters of the
//  * GRU layer work.
//  */
// TEST_CASE("GRULayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order : inSize, outSize, rho.
//   GRU<> layer1(1, 2, 3);
//   GRU<> layer2(1, 2, 4);

//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InSize() == 1);
//   REQUIRE(layer1.OutSize() == 2);
//   REQUIRE(layer1.Rho() == 3);

//   // Now modify the parameters to match the second layer.
//   layer1.Rho() = 4;

//   // Now ensure all the results are the same.
//   REQUIRE(layer1.InSize() == layer2.InSize());
//   REQUIRE(layer1.OutSize() == layer2.OutSize());
//   REQUIRE(layer1.Rho() == layer2.Rho());
// }

// /**
//  * Check if the gradients computed by GRU cell are close enough to the
//  * approximation of the gradients.
//  */
// TEST_CASE("GradientGRULayerTest", "[ANNLayerTest]")
// {
//   // GRU function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(1, 1, 5)),
//         target(arma::ones(1, 1, 5))
//     {
//       const size_t rho = 5;

//       model = new RNN<NegativeLogLikelihood>(rho);
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(1, 10);
//       model->Add<GRU<> >(10, 3, rho);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       arma::mat output;
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     RNN<NegativeLogLikelihood>* model;
//     arma::cube input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

// /**
//  * GRU layer manual forward test.
//  */
// TEST_CASE("ForwardGRULayerTest", "[ANNLayerTest]")
// {
//   // This will make it easier to clean memory later.
//   GRU<>* gruAlloc = new GRU<>(3, 3, 5);
//   GRU<>& gru = *gruAlloc;

//   // Initialize the weights to all ones.
//   NetworkInitialization<ConstInitialization>
//     networkInit(ConstInitialization(1));
//   networkInit.Initialize(gru.Model(), gru.Parameters());

//   // Provide input of all ones.
//   arma::mat input = arma::ones(3, 1);
//   arma::mat output;

//   gru.Forward(input, output);

//   // Compute the z_t gate output.
//   arma::mat expectedOutput = arma::ones(3, 1);
//   expectedOutput *= -4;
//   expectedOutput = arma::exp(expectedOutput);
//   expectedOutput = arma::ones(3, 1) / (arma::ones(3, 1) + expectedOutput);
//   expectedOutput = (arma::ones(3, 1)  - expectedOutput) % expectedOutput;

//   // For the first input the output should be equal to the output of
//   // gate z_t as the previous output fed to the cell is all zeros.
//   REQUIRE(arma::as_scalar(arma::trans(output) * expectedOutput) <= 1e-2);

//   expectedOutput = output;

//   gru.Forward(input, output);

//   double s = arma::as_scalar(arma::sum(expectedOutput));

//   // Compute the value of z_t gate for the second input.
//   arma::mat z_t = arma::ones(3, 1);
//   z_t *= -(s + 4);
//   z_t = arma::exp(z_t);
//   z_t = arma::ones(3, 1) / (arma::ones(3, 1) + z_t);

//   // Compute the value of o_t gate for the second input.
//   arma::mat o_t = arma::ones(3, 1);
//   o_t *= -(arma::as_scalar(arma::sum(expectedOutput % z_t)) + 4);
//   o_t = arma::exp(o_t);
//   o_t = arma::ones(3, 1) / (arma::ones(3, 1) + o_t);

//   // Expected output for the second input.
//   expectedOutput = z_t % expectedOutput + (arma::ones(3, 1) - z_t) % o_t;

//   REQUIRE(arma::as_scalar(arma::trans(output) * expectedOutput) <= 1e-2);

//   LayerTypes<> layer(gruAlloc);
//   boost::apply_visitor(DeleteVisitor(), layer);
// }

/**
 * Simple add merge module test.
 */
// TEST_CASE("SimpleAddMergeLayerTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta;
//   input = arma::ones(10, 1);
//
//   for (size_t i = 0; i < 5; ++i)
//   {
//     AddMerge<> module(false, false);
//     const size_t numMergeModules = RandInt(2, 10);
//     for (size_t m = 0; m < numMergeModules; ++m)
//     {
//       IdentityLayer<> identityLayer;
//       identityLayer.Forward(input, identityLayer.OutputParameter());
//
//       module.Add<IdentityLayer<> >(identityLayer);
//     }
//
//     // Test the Forward function.
//     module.Forward(input, output);
//     REQUIRE(10 * numMergeModules == arma::accu(output));
//
//     // Test the Backward function.
//     module.Backward(input, output, delta);
//     REQUIRE(arma::accu(output) == arma::accu(delta));
//   }
// }

/**
 * Test the LSTM layer with a user defined rho parameter and without.
 */
// TEST_CASE("LSTMRrhoTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5;
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::zeros(1, 1, 5);
//   RandomInitialization init(0.5, 0.5);
//
//   // Create model with user defined rho parameter.
//   RNN<NegativeLogLikelihood, RandomInitialization> modelA(
//       rho, false, NegativeLogLikelihood(), init);
//   modelA.Add<IdentityLayer<> >();
//   modelA.Add<Linear<> >(1, 10);
//
//   // Use LSTM layer with rho.
//   modelA.Add<LSTM<> >(10, 3, rho);
//   modelA.Add<LogSoftMax<> >();
//
//   // Create model without user defined rho parameter.
//   RNN<NegativeLogLikelihood> modelB(
//       rho, false, NegativeLogLikelihood(), init);
//   modelB.Add<IdentityLayer<> >();
//   modelB.Add<Linear<> >(1, 10);
//
//   // Use LSTM layer with rho = MAXSIZE.
//   modelB.Add<LSTM<> >(10, 3);
//   modelB.Add<LogSoftMax<> >();
//
//   ens::StandardSGD opt(0.1, 1, 5, -100, false);
//   modelA.Train(input, target, opt);
//   modelB.Train(input, target, opt);
//
//   CheckMatrices(modelB.Parameters(), modelA.Parameters());
// }

/**
 * LSTM layer numerical gradient test.
 */
// TEST_CASE("GradientLSTMLayerTest", "[ANNLayerTest]")
// {
//   // LSTM function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(1, 1, 5)),
//         target(arma::zeros(1, 1, 5))
//     {
//       const size_t rho = 5;
//
//       model = new RNN<NegativeLogLikelihood>(rho);
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(1, 10);
//       model->Add<LSTM<> >(10, 3, rho);
//       model->Add<LogSoftMax<> >();
//     }
//
//     ~GradientFunction()
//     {
//       delete model;
//     }
//
//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }
//
//     arma::mat& Parameters() { return model->Parameters(); }
//
//     RNN<NegativeLogLikelihood>* model;
//     arma::cube input, target;
//   } function;
//
//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

/**
 * Test that the functions that can modify and access the parameters of the
 * LSTM layer work.
 */
// TEST_CASE("LSTMLayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order : inSize, outSize, rho.
//   LSTM<> layer1(1, 2, 3);
//   LSTM<> layer2(1, 2, 4);
//
//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InSize() == 1);
//   REQUIRE(layer1.OutSize() == 2);
//   REQUIRE(layer1.Rho() == 3);
//
//   // Now modify the parameters to match the second layer.
//   layer1.Rho() = 4;
//
//   // Now ensure all the results are the same.
//   REQUIRE(layer1.InSize() == layer2.InSize());
//   REQUIRE(layer1.OutSize() == layer2.OutSize());
//   REQUIRE(layer1.Rho() == layer2.Rho());
// }

/**
 * Test the FastLSTM layer with a user defined rho parameter and without.
 */
// TEST_CASE("FastLSTMRrhoTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5;
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::zeros(1, 1, 5);
//   RandomInitialization init(0.5, 0.5);
//
//   // Create model with user defined rho parameter.
//   RNN<NegativeLogLikelihood, RandomInitialization> modelA(
//       rho, false, NegativeLogLikelihood(), init);
//   modelA.Add<IdentityLayer<> >();
//   modelA.Add<Linear<> >(1, 10);
//
//   // Use FastLSTM layer with rho.
//   modelA.Add<FastLSTM<> >(10, 3, rho);
//   modelA.Add<LogSoftMax<> >();
//
//   // Create model without user defined rho parameter.
//   RNN<NegativeLogLikelihood> modelB(
//       rho, false, NegativeLogLikelihood(), init);
//   modelB.Add<IdentityLayer<> >();
//   modelB.Add<Linear<> >(1, 10);
//
//   // Use FastLSTM layer with rho = MAXSIZE.
//   modelB.Add<FastLSTM<> >(10, 3);
//   modelB.Add<LogSoftMax<> >();
//
//   ens::StandardSGD opt(0.1, 1, 5, -100, false);
//   modelA.Train(input, target, opt);
//   modelB.Train(input, target, opt);
//
//   CheckMatrices(modelB.Parameters(), modelA.Parameters());
// }

/**
 * FastLSTM layer numerical gradient test.
 */
// TEST_CASE("GradientFastLSTMLayerTest", "[ANNLayerTest]")
// {
//   // Fast LSTM function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(1, 1, 5)),
//         target(arma::zeros(1, 1, 5))
//     {
//       const size_t rho = 5;
//
//       model = new RNN<NegativeLogLikelihood>(rho);
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(1, 10);
//       model->Add<FastLSTM<> >(10, 3, rho);
//       model->Add<LogSoftMax<> >();
//     }
//
//     ~GradientFunction()
//     {
//       delete model;
//     }
//
//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }
//
//     arma::mat& Parameters() { return model->Parameters(); }
//
//     RNN<NegativeLogLikelihood>* model;
//     arma::cube input, target;
//   } function;
//
//   // The threshold should be << 0.1 but since the Fast LSTM layer uses an
//   // approximation of the sigmoid function the estimated gradient is not
//   // correct.
//   REQUIRE(CheckGradient(function) <= 0.2);
// }

/**
 * Test that the functions that can modify and access the parameters of the
 * Fast LSTM layer work.
 */
// TEST_CASE("FastLSTMLayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order : inSize, outSize, rho.
//   FastLSTM<> layer1(1, 2, 3);
//   FastLSTM<> layer2(1, 2, 4);
//
//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InSize() == 1);
//   REQUIRE(layer1.OutSize() == 2);
//   REQUIRE(layer1.Rho() == 3);
//
//   // Now modify the parameters to match the second layer.
//   layer1.Rho() = 4;
//
//   // Now ensure all the results are the same.
//   REQUIRE(layer1.InSize() == layer2.InSize());
//   REQUIRE(layer1.OutSize() == layer2.OutSize());
//   REQUIRE(layer1.Rho() == layer2.Rho());
// }

/**
 * Check whether copying and moving network with FastLSTM is working or not.
 */
// TEST_CASE("CheckCopyMoveFastLSTMTest", "[ANNLayerTest]")
// {
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::ones(1, 1, 5);
//   const size_t rho = 5;
//
//   RNN<NegativeLogLikelihood> *model1 =
//       new RNN<NegativeLogLikelihood>(rho);
//   model1->ResetData(input, target);
//   model1->Add<IdentityLayer<> >();
//   model1->Add<Linear<> >(1, 10);
//   model1->Add<FastLSTM<> >(10, 3, rho);
//   model1->Add<LogSoftMax<> >();
//
//   RNN<NegativeLogLikelihood> *model2 =
//      new RNN<NegativeLogLikelihood>(rho);
//   model2->ResetData(input, target);
//   model2->Add<IdentityLayer<> >();
//   model2->Add<Linear<> >(1, 10);
//   model2->Add<FastLSTM<> >(10, 3, rho);
//   model2->Add<LogSoftMax<> >();
//
//   // Check whether copy constructor is working or not.
//   CheckRNNCopyFunction<>(model1, input, target, 1);
//
//   // Check whether move constructor is working or not.
//   CheckRNNMoveFunction<>(model2, input, target, 1);
// }

/**
 * Check whether copying and moving network with LSTM is working or not.
 */
// TEST_CASE("CheckCopyMoveLSTMTest", "[ANNLayerTest]")
// {
//   arma::cube input = arma::randu(1, 1, 5);
//   arma::cube target = arma::ones(1, 1, 5);
//   const size_t rho = 5;
//
//   RNN<NegativeLogLikelihood> *model1 =
//       new RNN<NegativeLogLikelihood>(rho);
//   model1->ResetData(input, target);
//   model1->Add<IdentityLayer<> >();
//   model1->Add<Linear<> >(1, 10);
//   model1->Add<LSTM<> >(10, 3, rho);
//   model1->Add<LogSoftMax<> >();
//
//   RNN<NegativeLogLikelihood> *model2 =
//      new RNN<NegativeLogLikelihood>(rho);
//   model2->ResetData(input, target);
//   model2->Add<IdentityLayer<> >();
//   model2->Add<Linear<> >(1, 10);
//   model2->Add<LSTM<> >(10, 3, rho);
//   model2->Add<LogSoftMax<> >();
//
//   // Check whether copy constructor is working or not.
//   CheckRNNCopyFunction<>(model1, input, target, 1);
//
//   // Check whether move constructor is working or not.
//   CheckRNNMoveFunction<>(model2, input, target, 1);
// }

/**
 * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
 * state. Besides output, the overloaded function provides read access to cell
 * state of the LSTM layer.
 */
// TEST_CASE("ReadCellStateParamLSTMLayerTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5, inputSize = 3, outputSize = 2;
//
//   // Provide input of all ones.
//   arma::cube input = arma::ones(inputSize, outputSize, rho);
//
//   arma::mat inputGate, forgetGate, outputGate, hidden;
//   arma::mat outLstm, cellLstm;
//
//   // LSTM layer.
//   LSTM<> lstm(inputSize, outputSize, rho);
//   lstm.Reset();
//   lstm.ResetCell(rho);
//
//   // Initialize the weights to all ones.
//   lstm.Parameters().ones();
//
//   arma::mat inputWeight = arma::ones(outputSize, inputSize);
//   arma::mat outputWeight = arma::ones(outputSize, outputSize);
//   arma::mat bias = arma::ones(outputSize, input.n_cols);
//   arma::mat cellCalc = arma::zeros(outputSize, input.n_cols);
//   arma::mat outCalc = arma::zeros(outputSize, input.n_cols);
//
//   for (size_t seqNum = 0; seqNum < rho; ++seqNum)
//   {
//       // Wrap a matrix around our data to avoid a copy.
//       arma::mat stepData(input.slice(seqNum).memptr(),
//           input.n_rows, input.n_cols, false, true);
//
//       // Apply Forward() on LSTM layer.
//       lstm.Forward(stepData, // Input.
//                    outLstm,  // Output.
//                    cellLstm, // Cell state.
//                    false); // Don't write into the cell state.
//
//       // Compute the value of cell state and output.
//       // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));
//
//       // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));
//
//       // z = tanh(W.dot(x) + W.dot(h) + b).
//       hidden = arma::tanh(inputWeight * stepData +
//                      outputWeight * outCalc + bias);
//
//       // c = f * c + i * z.
//       cellCalc = forgetGate % cellCalc + inputGate % hidden;
//
//       // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));
//
//       // h = o * tanh(c).
//       outCalc = outputGate % arma::tanh(cellCalc);
//
//       CheckMatrices(outLstm, outCalc, 1e-12);
//       CheckMatrices(cellLstm, cellCalc, 1e-12);
//   }
// }

/**
 * Testing the overloaded Forward() of the LSTM layer, for retrieving the cell
 * state. Besides output, the overloaded function provides write access to cell
 * state of the LSTM layer.
 */
// TEST_CASE("WriteCellStateParamLSTMLayerTest", "[ANNLayerTest]")
// {
//   const size_t rho = 5, inputSize = 3, outputSize = 2;
//
//   // Provide input of all ones.
//   arma::cube input = arma::ones(inputSize, outputSize, rho);
//
//   arma::mat inputGate, forgetGate, outputGate, hidden;
//   arma::mat outLstm, cellLstm;
//   arma::mat cellCalc;
//
//   // LSTM layer.
//   LSTM<> lstm(inputSize, outputSize, rho);
//   lstm.Reset();
//   lstm.ResetCell(rho);
//
//   // Initialize the weights to all ones.
//   lstm.Parameters().ones();
//
//   arma::mat inputWeight = arma::ones(outputSize, inputSize);
//   arma::mat outputWeight = arma::ones(outputSize, outputSize);
//   arma::mat bias = arma::ones(outputSize, input.n_cols);
//   arma::mat outCalc = arma::zeros(outputSize, input.n_cols);
//
//   for (size_t seqNum = 0; seqNum < rho; ++seqNum)
//   {
//       // Wrap a matrix around our data to avoid a copy.
//       arma::mat stepData(input.slice(seqNum).memptr(),
//           input.n_rows, input.n_cols, false, true);
//
//       if (cellLstm.is_empty())
//       {
//         // Set the cell state to zeros.
//         cellLstm = arma::zeros(outputSize, input.n_cols);
//         cellCalc = arma::zeros(outputSize, input.n_cols);
//       }
//       else
//       {
//         // Set the cell state to zeros.
//         cellLstm = arma::zeros(cellLstm.n_rows, cellLstm.n_cols);
//         cellCalc = arma::zeros(cellCalc.n_rows, cellCalc.n_cols);
//       }
//
//       // Apply Forward() on the LSTM layer.
//       lstm.Forward(stepData, // Input.
//                    outLstm,  // Output.
//                    cellLstm, // Cell state.
//                    true);  // Write into cell state.
//
//       // Compute the value of cell state and output.
//       // i = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       inputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));
//
//       // f = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       forgetGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));
//
//       // z = tanh(W.dot(x) + W.dot(h) + b).
//       hidden = arma::tanh(inputWeight * stepData +
//                      outputWeight * outCalc + bias);
//
//       // c = f * c + i * z.
//       cellCalc = forgetGate % cellCalc + inputGate % hidden;
//
//       // o = sigmoid(W.dot(x) + W.dot(h) + W.dot(c) + b).
//       outputGate = 1.0 /(1 + arma::exp(-(inputWeight * stepData +
//           outputWeight * outCalc + outputWeight % cellCalc + bias)));
//
//       // h = o * tanh(c).
//       outCalc = outputGate % arma::tanh(cellCalc);
//
//       CheckMatrices(outLstm, outCalc, 1e-12);
//       CheckMatrices(cellLstm, cellCalc, 1e-12);
//   }
//
//   // Attempting to write empty matrix into cell state.
//   lstm.Reset();
//   lstm.ResetCell(rho);
//   arma::mat stepData(input.slice(0).memptr(),
//       input.n_rows, input.n_cols, false, true);
//
//   lstm.Forward(stepData, // Input.
//                outLstm,  // Output.
//                cellLstm, // Cell state.
//                true); // Write into cell state.
//
//   for (size_t seqNum = 1; seqNum < rho; ++seqNum)
//   {
//     arma::mat empty;
//     // Should throw error.
//     REQUIRE_THROWS_AS(lstm.Forward(stepData, // Input.
//                                    outLstm,  // Output.
//                                    empty, // Cell state.
//                                    true),  // Write into cell state.
//                                    std::runtime_error);
//   }
// }

/**
 * Test that the functions that can modify and access the parameters of the
 * GRU layer work.
 */
// TEST_CASE("GRULayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order : inSize, outSize, rho.
//   GRU<> layer1(1, 2, 3);
//   GRU<> layer2(1, 2, 4);
//
//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InSize() == 1);
//   REQUIRE(layer1.OutSize() == 2);
//   REQUIRE(layer1.Rho() == 3);
//
//   // Now modify the parameters to match the second layer.
//   layer1.Rho() = 4;
//
//   // Now ensure all the results are the same.
//   REQUIRE(layer1.InSize() == layer2.InSize());
//   REQUIRE(layer1.OutSize() == layer2.OutSize());
//   REQUIRE(layer1.Rho() == layer2.Rho());
// }

/**
 * Check if the gradients computed by GRU cell are close enough to the
 * approximation of the gradients.
 */
// TEST_CASE("GradientGRULayerTest", "[ANNLayerTest]")
// {
//   // GRU function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(1, 1, 5)),
//         target(arma::zeros(1, 1, 5))
//     {
//       const size_t rho = 5;
//
//       model = new RNN<NegativeLogLikelihood>(rho);
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(1, 10);
//       model->Add<GRU<> >(10, 3, rho);
//       model->Add<LogSoftMax<> >();
//     }
//
//     ~GradientFunction()
//     {
//       delete model;
//     }
//
//     double Gradient(arma::mat& gradient) const
//     {
//       arma::mat output;
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }
//
//     arma::mat& Parameters() { return model->Parameters(); }
//
//     RNN<NegativeLogLikelihood>* model;
//     arma::cube input, target;
//   } function;
//
//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

/**
 * GRU layer manual forward test.
 */
// TEST_CASE("ForwardGRULayerTest", "[ANNLayerTest]")
// {
//   // This will make it easier to clean memory later.
//   GRU<>* gruAlloc = new GRU<>(3, 3, 5);
//   GRU<>& gru = *gruAlloc;
//
//   // Initialize the weights to all ones.
//   NetworkInitialization<ConstInitialization>
//     networkInit(ConstInitialization(1));
//   networkInit.Initialize(gru.Model(), gru.Parameters());
//
//   // Provide input of all ones.
//   arma::mat input = arma::ones(3, 1);
//   arma::mat output;
//
//   gru.Forward(input, output);
//
//   // Compute the z_t gate output.
//   arma::mat expectedOutput = arma::ones(3, 1);
//   expectedOutput *= -4;
//   expectedOutput = arma::exp(expectedOutput);
//   expectedOutput = arma::ones(3, 1) / (arma::ones(3, 1) + expectedOutput);
//   expectedOutput = (arma::ones(3, 1)  - expectedOutput) % expectedOutput;
//
//   // For the first input the output should be equal to the output of
//   // gate z_t as the previous output fed to the cell is all zeros.
//   REQUIRE(arma::as_scalar(arma::trans(output) * expectedOutput) <= 1e-2);
//
//   expectedOutput = output;
//
//   gru.Forward(input, output);
//
//   double s = arma::as_scalar(arma::sum(expectedOutput));
//
//   // Compute the value of z_t gate for the second input.
//   arma::mat z_t = arma::ones(3, 1);
//   z_t *= -(s + 4);
//   z_t = arma::exp(z_t);
//   z_t = arma::ones(3, 1) / (arma::ones(3, 1) + z_t);
//
//   // Compute the value of o_t gate for the second input.
//   arma::mat o_t = arma::ones(3, 1);
//   o_t *= -(arma::as_scalar(arma::sum(expectedOutput % z_t)) + 4);
//   o_t = arma::exp(o_t);
//   o_t = arma::ones(3, 1) / (arma::ones(3, 1) + o_t);
//
//   // Expected output for the second input.
//   expectedOutput = z_t % expectedOutput + (arma::ones(3, 1) - z_t) % o_t;
//
//   REQUIRE(arma::as_scalar(arma::trans(output) * expectedOutput) <= 1e-2);
//
//   LayerTypes<> layer(gruAlloc);
//   boost::apply_visitor(DeleteVisitor(), layer);
// }

/**
 * Simple concat module test.
 */
TEST_CASE("SimpleConcatLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta, error;

  Linear* moduleA = new Linear(10);
  moduleA->InputDimensions() = std::vector<size_t>({ 10 });
  moduleA->ComputeOutputDimensions();
  arma::mat weightsA(moduleA->WeightSize(), 1);
  moduleA->SetWeights((double*) weightsA.memptr());
  moduleA->Parameters().randu();

  Linear* moduleB = new Linear(10);
  moduleB->InputDimensions() = std::vector<size_t>({ 10 });
  moduleB->ComputeOutputDimensions();
  arma::mat weightsB(moduleB->WeightSize(), 1);
  moduleB->SetWeights((double*) weightsB.memptr());
  moduleB->Parameters().randu();

  Concat module;
  module.Add(moduleA);
  module.Add(moduleB);
  module.InputDimensions() = std::vector<size_t>({ 10 });
  module.ComputeOutputDimensions();

  // Test the Forward function.
  input = arma::zeros(10, 1);
  output.set_size(module.OutputSize(), 1);
  module.Forward(input, output);

  const double sumModuleA = arma::accu(
      moduleA->Parameters().submat(
      100, 0, moduleA->Parameters().n_elem - 1, 0));
  const double sumModuleB = arma::accu(
      moduleB->Parameters().submat(
      100, 0, moduleB->Parameters().n_elem - 1, 0));
  REQUIRE(sumModuleA + sumModuleB ==
      Approx(arma::accu(output.col(0))).epsilon(1e-5));

  // Test the Backward function.
  error = arma::zeros(20, 1);
  delta.set_size(input.n_rows, input.n_cols);
  module.Backward(input, error, delta);
  REQUIRE(arma::accu(delta) == 0);
}

/**
 * Test to check Concat layer along different axes.
 */
TEST_CASE("ConcatAlongAxisTest", "[ANNLayerTest]")
{
  arma::mat output, input, error, outputA, outputB;
  size_t inputWidth = 4, inputHeight = 4, inputChannel = 2;
  size_t outputWidth, outputHeight, outputChannel = 2;
  size_t kW = 3, kH = 3;
  size_t batch = 1;

  // Using Convolution<> layer as inout to Concat<> layer.
  // Compute the output shape of convolution layer.
  outputWidth  = (inputWidth - kW) + 1;
  outputHeight = (inputHeight - kH) + 1;

  input = arma::ones(inputWidth * inputHeight * inputChannel, batch);

  Convolution* moduleA = new Convolution(outputChannel, kW, kH, 1, 1, 0, 0);
  Convolution* moduleB = new Convolution(outputChannel, kW, kH, 1, 1, 0, 0);

  moduleA->InputDimensions() = std::vector<size_t>({ inputWidth, inputHeight });
  moduleA->ComputeOutputDimensions();
  arma::mat weightsA(moduleA->WeightSize(), 1);
  moduleA->SetWeights((double*) weightsA.memptr());
  moduleA->Parameters().randu();

  moduleB->InputDimensions() = std::vector<size_t>({ inputWidth, inputHeight });
  moduleB->ComputeOutputDimensions();
  arma::mat weightsB(moduleB->WeightSize(), 1);
  moduleB->SetWeights((double*) weightsB.memptr());
  moduleB->Parameters().randu();

  // Compute output of each layer.
  outputA.set_size(moduleA->OutputSize(), 1);
  outputB.set_size(moduleB->OutputSize(), 1);
  moduleA->Forward(input, outputA);
  moduleB->Forward(input, outputB);

  arma::cube A(outputA.memptr(), outputWidth, outputHeight, outputChannel);
  arma::cube B(outputB.memptr(), outputWidth, outputHeight, outputChannel);

  error = arma::ones(outputWidth * outputHeight * outputChannel * 2, 1);

  for (size_t axis = 0; axis < 3; ++axis)
  {
    size_t x = 1, y = 1, z = 1;
    arma::cube calculatedOut;
    if (axis == 0)
    {
      calculatedOut.set_size(2 * outputWidth, outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
        arma::mat aMat = A.slice(i);
        arma::mat bMat = B.slice(i);
        calculatedOut.slice(i) = arma::join_cols(aMat, bMat);
      }
      x = 2;
    }
    if (axis == 1)
    {
      calculatedOut.set_size(outputWidth, 2 * outputHeight, outputChannel);
      for (size_t i = 0; i < A.n_slices; ++i)
      {
        arma::mat aMat = A.slice(i);
        arma::mat bMat = B.slice(i);
        calculatedOut.slice(i) = arma::join_rows(aMat, bMat);
      }
      y = 2;
    }
    if (axis == 2)
    {
      calculatedOut = arma::join_slices(A, B);
      z = 2;
    }

    // Compute output of Concat<> layer.
    Concat module(axis);
    module.Add(moduleA);
    module.Add(moduleB);
    module.InputDimensions() = std::vector<size_t>({ inputWidth, inputHeight });
    module.ComputeOutputDimensions();
    output.set_size(module.OutputSize(), 1);
    module.Forward(input, output);
    arma::cube concatOut(output.memptr(), x * outputWidth,
        y * outputHeight, z * outputChannel);

    // Verify if the output reshaped to cubes are similar.
    CheckMatrices(concatOut, calculatedOut, 1e-12);

    // Ensure that the child layers don't get deleted when `module` is
    // deallocated.
    module.Network().clear();
  }

  delete moduleA;
  delete moduleB;
}

/**
 * Test that the function that can access the axis parameter of the
 * Concat layer works.
 */
TEST_CASE("ConcatLayerParametersTest", "[ANNLayerTest]")
{
  Concat layer(2);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.Axis() == 2);
}

/**
 * Concat layer numerical gradient test.
 */
TEST_CASE("GradientConcatLayerTest", "[ANNLayerTest]")
{
  // Concat function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(10);

      concat = new Concat();
      concat->Add<Linear>(5);
      concat->Add<Linear>(5);
      model->Add(concat);
      model->Add<Linear>(2);

      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    Concat* concat;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/**
 * Simple concatenate module test.
 */
TEST_CASE("SimpleConcatenateLayerTest", "[ANNLayerTest]")
{
  arma::mat input = arma::ones(5, 1);
  arma::mat output, delta;

  Concatenate module;
  module.Concat() = arma::ones(5, 1) * 0.5;
  module.InputDimensions() = std::vector<size_t>({ 5 });
  module.ComputeOutputDimensions();

  // Test the Forward function.
  output.set_size(module.OutputSize(), 1);
  module.Forward(input, output);

  REQUIRE(arma::accu(output) == 7.5);

  // Test the Backward function.
  delta.set_size(5, 1);
  module.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 5);
}

/**
 * Concatenate layer numerical gradient test.
 */
TEST_CASE("GradientConcatenateLayerTest", "[ANNLayerTest]")
{
  // Concatenate function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<Linear>(5);

      arma::mat concat = arma::ones(5, 1);
      // concatenate = new Concatenate();
      // concatenate->Concat() = concat;
      // model->Add(concatenate);
      model->Add<Concatenate>(concat);

      model->Add<Linear>(5);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    Concatenate* concatenate;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/**
 * Simple lookup module test.
 *
TEST_CASE("SimpleLookupLayerTest", "[ANNLayerTest]")
{
  const size_t vocabSize = 10;
  const size_t embeddingSize = 2;
  const size_t seqLength = 3;
  const size_t batchSize = 4;

  arma::mat output, input, gy, g, gradient;

  Lookup module(vocabSize, embeddingSize);
  module.Parameters().randu();

  // Test the Forward function.
  input = arma::zeros(seqLength, batchSize);
  for (size_t i = 0; i < input.n_elem; ++i)
  {
    int token = RandInt(1, vocabSize);
    input(i) = token;
  }

  module.Forward(input, output);
  for (size_t i = 0; i < batchSize; ++i)
  {
    // The Lookup module uses index - 1 for the cols.
    const double outputSum = arma::accu(module.Parameters().cols(
        arma::conv_to<arma::uvec>::from(input.col(i)) - 1));

    REQUIRE(std::fabs(outputSum - arma::accu(output.col(i))) <= 1e-5);
  }

  // Test the Gradient function.
  arma::mat error = 0.01 * arma::randu(embeddingSize * seqLength, batchSize);
  module.Gradient(input, error, gradient);

  REQUIRE(std::fabs(arma::accu(error) - arma::accu(gradient)) <= 1e-07);
}
*/

/**
 * Lookup layer numerical gradient test.
 *
TEST_CASE("GradientLookupLayerTest", "[ANNLayerTest]")
{
  // Lookup function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction()
    {
      input.set_size(seqLength, batchSize);
      for (size_t i = 0; i < input.n_elem; ++i)
      {
        input(i) = RandInt(1, vocabSize);
      }
      target = arma::zeros(vocabSize, batchSize);
      for (size_t i = 0; i < batchSize; ++i)
      {
        const size_t targetWord = RandInt(1, vocabSize);
        target(targetWord, i) = 1;
      }

      model = new FFN<BCELoss<>, GlorotInitialization>(BCELoss<>(1e-10, false));
      model->ResetData(input, target);
      model->Add<Lookup>(vocabSize, embeddingSize);
      model->Add<Linear>(embeddingSize * seqLength, vocabSize);
      model->Add<Softmax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, batchSize);
      model->Gradient(model->Parameters(), 0, gradient, batchSize);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<BCELoss<>, GlorotInitialization>* model;
    arma::mat input, target;

    const size_t seqLength = 10;
    const size_t embeddingSize = 8;
    const size_t vocabSize = 20;
    const size_t batchSize = 4;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-6);
}
*/

/**
 * Test that the functions that can access the parameters of the
 * Lookup layer work.
 *
TEST_CASE("LookupLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : vocabSize, embedingSize.
  Lookup layer(100, 8);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.VocabSize() == 100);
  REQUIRE(layer.EmbeddingSize() == 8);
}
*/

/**
 * Simple Softmax module test.
 */
TEST_CASE("SimpleSoftmaxLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output, gy, g;
  Softmax module;

  // Test the forward function.
  input = arma::mat("1.7; 3.6");
  module.Forward(input, output);
  REQUIRE(arma::accu(arma::abs(arma::mat("0.130108; 0.869892") - output)) ==
      Approx(0.0).margin(1e-4));

  // Test the backward function.
  gy = arma::zeros(input.n_rows, input.n_cols);
  gy(0) = 1;
  module.Backward(output, gy, g);
  REQUIRE(arma::accu(arma::abs(arma::mat("0.11318; -0.11318") - g)) ==
      Approx(0.0).margin(1e-04));
}

/**
 * Softmax layer numerical gradient test.
 */
TEST_CASE("GradientSoftmaxTest", "[ANNLayerTest]")
{
  // Softmax function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("1; 0"))
    {
      model = new FFN<MeanSquaredError, RandomInitialization>;
      model->ResetData(input, target);
      model->Add<Linear>(10);
      model->Add<ReLU>();
      model->Add<Linear>(2);
      model->Add<Softmax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<MeanSquaredError>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}

/**
 * Simple test for the NearestInterpolation layer
 *
TEST_CASE("SimpleNearestInterpolationLayerTest", "[ANNLayerTest]")
{
  // Tested output against torch.nn.Upsample(mode="nearest").
  arma::mat input, output, unzoomedOutput, expectedOutput;
  size_t inRowSize = 2;
  size_t inColSize = 2;
  size_t outRowSize = 5;
  size_t outColSize = 7;
  size_t depth = 1;
  input.zeros(inRowSize * inColSize * depth, 1);
  input[0] = 1.0;
  input[1] = 3.0;
  input[2] = 2.0;
  input[3] = 4.0;
  NearestInterpolation<> layer(inRowSize, inColSize, outRowSize,
                               outColSize, depth);

  expectedOutput << 1.0000 << 1.0000 << 1.0000 << 1.0000 << 2.0000
                 << 2.0000 << 2.0000 << arma::endr
                 << 1.0000 << 1.0000 << 1.0000 << 1.0000 << 2.0000
                 << 2.0000 << 2.0000 << arma::endr
                 << 1.0000 << 1.0000 << 1.0000 << 1.0000 << 2.0000
                 << 2.0000 << 2.0000 << arma::endr
                 << 3.0000 << 3.0000 << 3.0000 << 3.0000 << 4.0000
                 << 4.0000 << 4.0000 << arma::endr
                 << 3.0000 << 3.0000 << 3.0000 << 3.0000 << 4.0000
                 << 4.0000 << 4.0000 << arma::endr;
  expectedOutput.reshape(35, 1);

  layer.Forward(input, output);
  CheckMatrices(output - expectedOutput,
                arma::zeros(output.n_rows), 1e-4);

  expectedOutput.clear();
  expectedOutput << 12.0000 << 18.0000 << arma::endr
                 << 24.0000 << 24.0000 << arma::endr;
  expectedOutput.reshape(4, 1);
  layer.Backward(output, output, unzoomedOutput);
  CheckMatrices(unzoomedOutput - expectedOutput,
      arma::zeros(input.n_rows), 1e-4);

  arma::mat input1, output1, unzoomedOutput1, expectedOutput1;
  inRowSize = 2;
  inColSize = 3;
  outRowSize = 17;
  outColSize = 23;
  input1 << 1 << 2 << 3 << arma::endr
         << 4 << 5 << 6 << arma::endr;
  input1.reshape(6, 1);
  NearestInterpolation<> layer1(inRowSize, inColSize, outRowSize,
                                outColSize, depth);

  layer1.Forward(input1, output1);
  layer1.Backward(output1, output1, unzoomedOutput1);

  REQUIRE(arma::accu(output1) - 1317.00 == Approx(0.0).margin(1e-05));
  REQUIRE(arma::accu(unzoomedOutput1) - 1317.00 ==
          Approx(0.0).margin(1e-05));
}
*/

/*
 * Simple test for the BilinearInterpolation layer
 *
TEST_CASE("SimpleBilinearInterpolationLayerTest", "[ANNLayerTest]")
{
  // Tested output against tensorflow.image.resize_bilinear()
  arma::mat input, output, unzoomedOutput, expectedOutput;
  size_t inRowSize = 2;
  size_t inColSize = 2;
  size_t outRowSize = 5;
  size_t outColSize = 5;
  size_t depth = 1;
  input.zeros(inRowSize * inColSize * depth, 1);
  input[0] = 1.0;
  input[1] = input[2] = 2.0;
  input[3] = 3.0;
  BilinearInterpolation layer(inRowSize, inColSize, outRowSize, outColSize,
      depth);
  expectedOutput = arma::mat("1.0000 1.4000 1.8000 2.0000 2.0000 \
      1.4000 1.8000 2.2000 2.4000 2.4000 \
      1.8000 2.2000 2.6000 2.8000 2.8000 \
      2.0000 2.4000 2.8000 3.0000 3.0000 \
      2.0000 2.4000 2.8000 3.0000 3.0000");
  expectedOutput.reshape(25, 1);
  layer.Forward(input, output);
  CheckMatrices(output - expectedOutput, arma::zeros(output.n_rows), 1e-12);

  expectedOutput = arma::mat("1.0000 1.9000 1.9000 2.8000");
  expectedOutput.reshape(4, 1);
  layer.Backward(output, output, unzoomedOutput);
  CheckMatrices(unzoomedOutput - expectedOutput,
      arma::zeros(input.n_rows), 1e-12);
}
*/

/**
 * Test that the functions that can modify and access the parameters of the
 * Bilinear Interpolation layer work.
 *
TEST_CASE("BilinearInterpolationLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : inRowSize, inColSize, outRowSize, outColSize, depth.
  BilinearInterpolation layer1(1, 2, 3, 4, 5);
  BilinearInterpolation layer2(2, 3, 4, 5, 6);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer1.InRowSize() == 1);
  REQUIRE(layer1.InColSize() == 2);
  REQUIRE(layer1.OutRowSize() == 3);
  REQUIRE(layer1.OutColSize() == 4);
  REQUIRE(layer1.InDepth() == 5);

  // Now modify the parameters to match the second layer.
  layer1.InRowSize() = 2;
  layer1.InColSize() = 3;
  layer1.OutRowSize() = 4;
  layer1.OutColSize() = 5;
  layer1.InDepth() = 6;

  // Now ensure all results are the same.
  REQUIRE(layer1.InRowSize() == layer2.InRowSize());
  REQUIRE(layer1.InColSize() == layer2.InColSize());
  REQUIRE(layer1.OutRowSize() == layer2.OutRowSize());
  REQUIRE(layer1.OutColSize() == layer2.OutColSize());
  REQUIRE(layer1.InDepth() == layer2.InDepth());
}
*/

/*
 * Simple test for the BicubicInterpolation layer.
 *
TEST_CASE("SimpleBicubicInterpolationLayerTest", "[ANNLayerTest]")
{
  // Tested output against torch.nn.Upsample(mode="bicubic").
  // Test case with square input with rectangular output.
  arma::mat input, output, unzoomedOutput, expectedOutput;
  size_t inRowSize = 2;
  size_t inColSize = 2;
  size_t outRowSize = 5;
  size_t outColSize = 7;
  size_t depth = 1;
  input.zeros(inRowSize * inColSize * depth, 1);

  input << 10 << 20 << arma::endr
        << 30 << 40 << arma::endr;
  input.reshape(4, 1);
  BicubicInterpolation<> layer(inRowSize, inColSize, outRowSize,
                               outColSize, depth);

  expectedOutput << 6.68803935860  <<  7.33308309038 <<  9.69733236152
                 << 12.79500000000 << 15.89266763848 << 18.25691690962
                 << 18.90196064140 << arma::endr
                 << 10.53303935860 << 11.17808309038 << 13.54233236152
                 << 16.64000000000 << 19.73766763848 << 22.10191690962
                 << 22.74696064140 << arma::endr
                 << 18.89303935860 << 19.53808309038 << 21.90233236152
                 << 25.00000000000 << 28.09766763848 << 30.46191690962
                 << 31.10696064140 << arma::endr
                 << 27.25303935860 << 27.89808309038 << 30.26233236152
                 << 33.36000000000 << 36.45766763848 << 38.82191690962
                 << 39.46696064140 << arma::endr
                 << 31.09803935860 << 31.74308309038 << 34.10733236152
                 << 37.20500000000 << 40.30266763848 << 42.66691690962
                 << 43.31196064140 << arma::endr;
  expectedOutput.reshape(35, 1);
  layer.Forward(input, output);

  CheckMatrices(output, expectedOutput, 1e-6);

  expectedOutput.clear();
  expectedOutput << 103.79040654914 << 180.51345595086 << arma::endr
                 << 256.98654404914 << 333.70959345086 << arma::endr;
  expectedOutput.reshape(4, 1);

  layer.Backward(output, output, unzoomedOutput);

  CheckMatrices(unzoomedOutput, expectedOutput, 1e-6);

  // Tested output against torch.nn.Upsample(mode="bicubic").
  // Test case with rectangular input with rectangular output.
  arma::mat input1, output1, unzoomedOutput1, expectedOutput1, expectedUnzoomed;

  inRowSize = 2;
  inColSize = 3;
  outRowSize = 5;
  outColSize = 7;
  depth = 1;
  input1.zeros(inRowSize * inColSize * depth, 1);

  input1 << 10 << 20 << 30 << arma::endr
         << 40 << 50 << 60 << arma::endr;
  input1.reshape(6, 1);

  BicubicInterpolation<> layer1(inRowSize, inColSize, outRowSize,
                                outColSize, depth);

  expectedOutput1 << 5.59920553936  << 7.77121720117  << 11.44468658892
                  << 16.69250000000 << 21.94031341108 << 25.61378279883
                  << 27.78579446064 << arma::endr
                  << 11.36670553936 << 13.53871720117 << 17.21218658892
                  << 22.46000000000 << 27.70781341108 << 31.38128279883
                  << 33.55329446064 << arma::endr
                  << 23.90670553936 << 26.07871720117 << 29.75218658892
                  << 35.00000000000 << 40.24781341108 << 43.92128279883
                  << 46.09329446064 << arma::endr
                  << 36.44670553936 << 38.61871720117 << 42.29218658892
                  << 47.54000000000 << 52.78781341108 << 56.46128279883
                  << 58.63329446064 << arma::endr
                  << 42.21420553936 << 44.38621720117 << 48.05968658892
                  << 53.30750000000 << 58.55531341108 << 62.22878279883
                  << 64.40079446064 << arma::endr;
  expectedOutput1.reshape(35, 1);
  layer1.Forward(input1, output1);

  CheckMatrices(output1, expectedOutput1, 1e-6);

  expectedUnzoomed << 67.65674505130  << 132.29729646501
                   << 182.75175223368 << arma::endr
                   << 218.01355388877 << 291.17209129009
                   << 333.10856107115 << arma::endr;
  expectedUnzoomed.reshape(6, 1);

  layer1.Backward(output1, output1, unzoomedOutput1);
  CheckMatrices(unzoomedOutput1, expectedUnzoomed, 1e-6);
}
*/

/**
 * VirtualBatchNorm layer numerical gradient test.
 *
TEST_CASE("GradientVirtualBatchNormTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randn(5, 256)),
        target(arma::zeros(1, 256))
    {
      arma::mat referenceBatch = arma::mat(input.memptr(), input.n_rows, 4);

      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<IdentityLayer>();
      model->Add<Linear>(5, 5);
      model->Add<VirtualBatchNorm>(referenceBatch, 5);
      model->Add<Linear>(5, 2);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 16, false);
      model->Gradient(model->Parameters(), 0, gradient, 16);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
*/

/**
 * Test that the functions that can modify and access the parameters of the
 * Virtual Batch Norm layer work.
 *
TEST_CASE("VirtualBatchNormLayerParametersTest", "[ANNLayerTest]")
{
  arma::mat input = arma::randn(5, 16);
  arma::mat referenceBatch = arma::mat(input.memptr(), input.n_rows, 4);

  // Parameter order : referenceBatch, size, eps.
  VirtualBatchNorm layer(referenceBatch, 5, 1e-3);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.InSize() == 5);
  REQUIRE(layer.Epsilon() == 1e-3);
}
*/

// /**
//  * MiniBatchDiscrimination layer numerical gradient test.
//  */
// TEST_CASE("MiniBatchDiscriminationTest", "[ANNLayerTest]")
// {
//   // Add function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randn(5, 4)),
//         target(arma::zeros(1, 4))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(5, 5);
//       model->Add<MiniBatchDiscrimination<> >(5, 10, 16);
//       model->Add<Linear<> >(10, 2);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       return model->EvaluateWithGradient(model->Parameters(), 0, gradient, 4);
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

/**
 * Simple Transposed Convolution layer test.
 *
TEST_CASE("SimpleTransposedConvolutionLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;

  TransposedConvolution module1(1, 1, 3, 3, 1, 1, 0, 0, 4, 4, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Parameters()(0) = 1.0;
  module1.Parameters()(8) = 2.0;
  module1.Reset();
  module1.Forward(input, output);
  // Value calculated using tensorflow.nn.conv2d_transpose()
  REQUIRE(arma::accu(output) == 360.0);

  // Test the backward function.
  module1.Backward(input, output, delta);
  // Value calculated using tensorflow.nn.conv2d()
  REQUIRE(arma::accu(delta) == 720.0);

  TransposedConvolution module2(1, 1, 4, 4, 1, 1, 1, 1, 5, 5, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module2.Parameters() = arma::mat(16 + 1, 1, arma::fill::zeros);
  module2.Parameters()(0) = 1.0;
  module2.Parameters()(3) = 1.0;
  module2.Parameters()(6) = 1.0;
  module2.Parameters()(9) = 1.0;
  module2.Parameters()(12) = 1.0;
  module2.Parameters()(15) = 2.0;
  module2.Reset();
  module2.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(arma::accu(output) == 1512.0);

  // Test the backward function.
  module2.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(arma::accu(delta) == 6504.0);

  TransposedConvolution module3(1, 1, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module3.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module3.Parameters()(1) = 2.0;
  module3.Parameters()(2) = 4.0;
  module3.Parameters()(3) = 3.0;
  module3.Parameters()(8) = 1.0;
  module3.Reset();
  module3.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(arma::accu(output) == 2370.0);

  // Test the backward function.
  module3.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(arma::accu(delta) == 19154.0);

  TransposedConvolution module4(1, 1, 3, 3, 1, 1, 0, 0, 5, 5, 7, 7);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module4.Parameters()(2) = 2.0;
  module4.Parameters()(4) = 4.0;
  module4.Parameters()(6) = 6.0;
  module4.Parameters()(8) = 8.0;
  module4.Reset();
  module4.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(arma::accu(output) == 6000.0);

  // Test the backward function.
  module4.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(arma::accu(delta) == 86208.0);

  TransposedConvolution module5(1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module5.Parameters()(2) = 8.0;
  module5.Parameters()(4) = 6.0;
  module5.Parameters()(6) = 4.0;
  module5.Parameters()(8) = 2.0;
  module5.Reset();
  module5.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(arma::accu(output) == 120.0);

  // Test the backward function.
  module5.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(arma::accu(delta) == 960.0);

  TransposedConvolution module6(1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 5, 5);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module6.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module6.Parameters()(0) = 8.0;
  module6.Parameters()(3) = 6.0;
  module6.Parameters()(6) = 2.0;
  module6.Parameters()(8) = 4.0;
  module6.Reset();
  module6.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(arma::accu(output) == 410.0);

  // Test the backward function.
  module6.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(arma::accu(delta) == 4444.0);

  TransposedConvolution module7(1, 1, 3, 3, 2, 2, 1, 1, 3, 3, 6, 6);
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module7.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module7.Parameters()(0) = 8.0;
  module7.Parameters()(2) = 6.0;
  module7.Parameters()(4) = 2.0;
  module7.Parameters()(8) = 4.0;
  module7.Reset();
  module7.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d()
  REQUIRE(arma::accu(output) == 606.0);

  module7.Backward(input, output, delta);
  // Value calculated using torch.nn.functional.conv2d()
  REQUIRE(arma::accu(delta) == 7732.0);
}
*/

/**
 * Transposed Convolution layer numerical gradient test.
 *
TEST_CASE("GradientTransposedConvolutionLayerTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  // To make this test robust, check it five times.
  bool pass = false;
  for (size_t trial = 0; trial < 5; trial++)
  {
    struct GradientFunction
    {
      GradientFunction() :
          input(arma::linspace<arma::colvec>(0, 35, 36)),
          target(arma::mat("0"))
      {
        model = new FFN<NegativeLogLikelihood, RandomInitialization>();
        model->ResetData(input, target);
        model->Add<TransposedConvolution>(1, 1, 3, 3, 2, 2, 1, 1, 6, 6, 12, 12);
        model->Add<LogSoftMax>();
      }

      ~GradientFunction()
      {
        delete model;
      }

      double Gradient(arma::mat& gradient) const
      {
        double error = model->Evaluate(model->Parameters(), 0, 1);
        model->Gradient(model->Parameters(), 0, gradient, 1);
        return error;
      }

      arma::mat& Parameters() { return model->Parameters(); }

      FFN<NegativeLogLikelihood, RandomInitialization>* model;
      arma::mat input, target;
    } function;

    if (CheckGradient(function) < 1e-3)
    {
      pass = true;
      break;
    }
  }
  REQUIRE(pass == true);
}
*/

/**
 * Simple MultiplyMerge module test.
 *
TEST_CASE("SimpleMultiplyMergeLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;
  input = arma::ones(10, 1);

  for (size_t i = 0; i < 5; ++i)
  {
    MultiplyMerge module(false, false);
    const size_t numMergeModules = RandInt(2, 10);
    for (size_t m = 0; m < numMergeModules; ++m)
    {
      IdentityLayer* identityLayer = new IdentityLayer();
      identityLayer->Forward(input, identityLayer->OutputParameter());

      module.Add(identityLayer);
    }

    // Test the Forward function.
    module.Forward(input, output);
    REQUIRE(10 == arma::accu(output));

    // Test the Backward function.
    module.Backward(input, output, delta);
    REQUIRE(arma::accu(output) == arma::accu(delta));
  }
}
*/

/**
 * Check whether copying and moving network with MultiplyMerge is working or
 * not.
 */
// TEST_CASE("CheckCopyMoveMultiplyMergeTest", "[ANNLayerTest]")
// {
//   arma::mat input(10, 1);
//   input.randu();
//
//   arma::mat output1;
//   arma::mat output2;
//   arma::mat output3;
//   arma::mat output4;
//
//   const size_t numMergeModules = RandInt(2, 10);
//
//   MultiplyMerge<> *module1 = new MultiplyMerge<>(true, false);
//   for (size_t m = 0; m < numMergeModules; ++m)
//   {
//     IdentityLayer<> identityLayer;
//     identityLayer.Forward(input, identityLayer.OutputParameter());
//
//     module1->Add<IdentityLayer<> >(identityLayer);
//   }
//
//   module1->Forward(input, output1);
//
//   MultiplyMerge<> module2 = *module1;
//   delete module1;
//
//   module2.Forward(input, output2);
//   CheckMatrices(output1, output2);
//
//   MultiplyMerge<> *module3 = new MultiplyMerge<>(true, false);
//   for (size_t m = 0; m < numMergeModules; ++m)
//   {
//     IdentityLayer<> identityLayer;
//     identityLayer.Forward(input, identityLayer.OutputParameter());
//
//     module3->Add<IdentityLayer<> >(identityLayer);
//   }
//   module3->Forward(input, output3);
//
//   MultiplyMerge<> module4(std::move(*module3));
//   delete module3;
//
//   module4.Forward(input, output4);
//   CheckMatrices(output3, output4);
// }

// /**
//  * Simple Atrous Convolution layer test.
//  */
// TEST_CASE("SimpleAtrousConvolutionLayerTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta;

//   AtrousConvolution<> module1(1, 1, 3, 3, 1, 1, 0, 0, 7, 7, 2, 2);
//   // Test the Forward function.
//   input = arma::linspace<arma::colvec>(0, 48, 49);
//   module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
//   module1.Parameters()(0) = 1.0;
//   module1.Parameters()(8) = 2.0;
//   module1.Reset();
//   module1.Forward(input, output);
//   // Value calculated using tensorflow.nn.atrous_conv2d()
//   REQUIRE(arma::accu(output) == 792.0);

//   // Test the Backward function.
//   module1.Backward(input, output, delta);
//   REQUIRE(arma::accu(delta) == 2376);

//   AtrousConvolution<> module2(1, 1, 3, 3, 2, 2, 0, 0, 7, 7, 2, 2);
//   // Test the forward function.
//   input = arma::linspace<arma::colvec>(0, 48, 49);
//   module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
//   module2.Parameters()(0) = 1.0;
//   module2.Parameters()(3) = 1.0;
//   module2.Parameters()(6) = 1.0;
//   module2.Reset();
//   module2.Forward(input, output);
//   // Value calculated using tensorflow.nn.conv2d()
//   REQUIRE(arma::accu(output) == 264.0);

//   // Test the backward function.
//   module2.Backward(input, output, delta);
//   REQUIRE(arma::accu(delta) == 792.0);
// }

// /**
//  * Atrous Convolution layer numerical gradient test.
//  */
// TEST_CASE("GradientAtrousConvolutionLayerTest", "[ANNLayerTest]")
// {
//   // Add function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::linspace<arma::colvec>(0, 35, 36)),
//         target(arma::mat("0"))
//     {
//       model = new FFN<NegativeLogLikelihood, RandomInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<AtrousConvolution<> >(1, 1, 3, 3, 1, 1, 0, 0, 6, 6, 2, 2);
//       model->Add<LogSoftMax<> >();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, RandomInitialization>* model;
//     arma::mat input, target;
//   } function;

//   // TODO: this tolerance seems far higher than necessary. The implementation
//   // should be checked.
//   REQUIRE(CheckGradient(function) <= 0.2);
// }

// /**
//  * Test the functions to access and modify the parameters of the
//  * AtrousConvolution layer.
//  */
// TEST_CASE("AtrousConvolutionLayerParametersTest", "[ANNLayerTest]")
// {
//   // Parameter order for the constructor: inSize, outSize, kW, kH, dW, dH, padW,
//   // padH, inputWidth, inputHeight, dilationW, dilationH, paddingType ("none").
//   AtrousConvolution<> layer1(1, 2, 3, 4, 5, 6, std::make_tuple(7, 8),
//       std::make_tuple(9, 10), 11, 12, 13, 14);
//   AtrousConvolution<> layer2(2, 3, 4, 5, 6, 7, std::make_tuple(8, 9),
//       std::make_tuple(10, 11), 12, 13, 14, 15);

//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer1.InputWidth() == 11);
//   REQUIRE(layer1.InputHeight() == 12);
//   REQUIRE(layer1.KernelWidth() == 3);
//   REQUIRE(layer1.KernelHeight() == 4);
//   REQUIRE(layer1.StrideWidth() == 5);
//   REQUIRE(layer1.StrideHeight() == 6);
//   REQUIRE(layer1.Padding().PadHTop() == 9);
//   REQUIRE(layer1.Padding().PadHBottom() == 10);
//   REQUIRE(layer1.Padding().PadWLeft() == 7);
//   REQUIRE(layer1.Padding().PadWRight() == 8);
//   REQUIRE(layer1.DilationWidth() == 13);
//   REQUIRE(layer1.DilationHeight() == 14);

//   // Now modify the parameters to match the second layer.
//   layer1.InputWidth() = 12;
//   layer1.InputHeight() = 13;
//   layer1.KernelWidth() = 4;
//   layer1.KernelHeight() = 5;
//   layer1.StrideWidth() = 6;
//   layer1.StrideHeight() = 7;
//   layer1.Padding().PadHTop() = 10;
//   layer1.Padding().PadHBottom() = 11;
//   layer1.Padding().PadWLeft() = 8;
//   layer1.Padding().PadWRight() = 9;
//   layer1.DilationWidth() = 14;
//   layer1.DilationHeight() = 15;

//   // Now ensure all results are the same.
//   REQUIRE(layer1.InputWidth() == layer2.InputWidth());
//   REQUIRE(layer1.InputHeight() == layer2.InputHeight());
//   REQUIRE(layer1.KernelWidth() == layer2.KernelWidth());
//   REQUIRE(layer1.KernelHeight() == layer2.KernelHeight());
//   REQUIRE(layer1.StrideWidth() == layer2.StrideWidth());
//   REQUIRE(layer1.StrideHeight() == layer2.StrideHeight());
//   REQUIRE(layer1.Padding().PadHTop() == layer2.Padding().PadHTop());
//   REQUIRE(layer1.Padding().PadHBottom() ==
//                       layer2.Padding().PadHBottom());
//   REQUIRE(layer1.Padding().PadWLeft() ==
//                       layer2.Padding().PadWLeft());
//   REQUIRE(layer1.Padding().PadWRight() ==
//                       layer2.Padding().PadWRight());
//   REQUIRE(layer1.DilationWidth() == layer2.DilationWidth());
//   REQUIRE(layer1.DilationHeight() == layer2.DilationHeight());
// }

// /**
//  * Test that the padding options are working correctly in Atrous Convolution
//  * layer.
//  */
// TEST_CASE("AtrousConvolutionLayerPaddingTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta;

//   // Check valid padding option.
//   AtrousConvolution<> module1(1, 1, 3, 3, 1, 1,
//       std::tuple<size_t, size_t>(1, 1), std::tuple<size_t, size_t>(1, 1), 7, 7,
//       2, 2, "valid");

//   // Test the Forward function.
//   input = arma::linspace<arma::colvec>(0, 48, 49);
//   module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
//   module1.Reset();
//   module1.Forward(input, output);

//   REQUIRE(arma::accu(output) == 0);
//   REQUIRE(output.n_rows == 9);
//   REQUIRE(output.n_cols == 1);

//   // Test the Backward function.
//   module1.Backward(input, output, delta);

//   // Check same padding option.
//   AtrousConvolution<> module2(1, 1, 3, 3, 1, 1,
//       std::tuple<size_t, size_t>(0, 0), std::tuple<size_t, size_t>(0, 0), 7, 7,
//       2, 2, "same");

//   // Test the forward function.
//   input = arma::linspace<arma::colvec>(0, 48, 49);
//   module2.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
//   module2.Reset();
//   module2.Forward(input, output);

//   REQUIRE(arma::accu(output) == 0);
//   REQUIRE(output.n_rows == 49);
//   REQUIRE(output.n_cols == 1);

//   // Test the backward function.
//   module2.Backward(input, output, delta);
// }

/**
 * Tests the GroupNorm layer.
 */
// TEST_CASE("GroupNormTest", "[ANNLayerTest]")
// {
//   arma::mat input, output, backwardOutput;
//   input = {
//     { 2, 0, 1 },
//     { 3, 1, 2 },
//     { 5, 1, 3 },
//     { 7, 2, 4 },
//     { 11, 3, 5 },
//     { 13, 5, 6 },
//     { 17, 8, 7 },
//     { 19, 13, 8 }
//   };
//
//   GroupNorm<> model(2, 4);
//   model.Reset();
//
//   model.Forward(input, output);
//   arma::mat result;
//   result = {
//     { -1.1717001972, -1.4142135482, -1.3416407811 },
//     { -0.6509445540, 0.0000000000 , -0.4472135937 },
//     { 0.3905667324 , 0.0000000000 , 0.4472135937  },
//     { 1.4320780188 , 1.4142135482 , 1.341640781   },
//     { -1.2649110634, -1.1283296293, -1.3416407811 },
//     { -0.6324555317, -0.5973509802, -0.4472135937 },
//     { 0.6324555317 , 0.1991169934 , 0.4472135937  },
//     { 1.2649110634 , 1.5265636161 , 1.3416407811  }
//   };
//
//   CheckMatrices(output, result, 1e-5);
// }

/**
 * GroupNorm layer numerical gradient test.
 */
// TEST_CASE("GradientGroupNormTest", "[ANNLayerTest]")
// {
//   // Add function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randn(10, 256)),
//         target(arma::zeros(1, 256))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer<> >();
//       model->Add<Linear<> >(10, 10);
//       model->Add<GroupNorm<> >(1, 10);
//       model->Add<Linear<> >(10, 2);
//       model->Add<LogSoftMax<> >();
//     }
//
//     ~GradientFunction()
//     {
//       delete model;
//     }
//
//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 256, false);
//       model->Gradient(model->Parameters(), 0, gradient, 256);
//       return error;
//     }
//
//     arma::mat& Parameters() { return model->Parameters(); }
//
//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     arma::mat input, target;
//   } function;
//
//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

/**
 * Tests the LayerNorm layer.
 *
TEST_CASE("LayerNormTest", "[ANNLayerTest]")
{
  arma::mat input, output;
  input = { { 5.1, 3.5 },
            { 4.9, 3.0 },
            { 4.7, 3.2 } };

  LayerNorm model(input.n_rows);
  model.Reset();

  model.Forward(input, output);
  arma::mat result;
  result = { { 1.2247, 1.2978 },
              { 0, -1.1355 },
              { -1.2247, -0.1622 } };

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.Mean();
  result = { 4.9000, 3.2333 };

  CheckMatrices(output, result, 1e-1);
  result.clear();

  output = model.Variance();
  result = { 0.0267, 0.0422 };

  CheckMatrices(output, result, 1e-1);
}
*/

/**
 * LayerNorm layer numerical gradient test.
 *
TEST_CASE("GradientLayerNormTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randn(10, 256)),
        target(arma::zeros(1, 256))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<IdentityLayer>();
      model->Add<Linear>(10, 10);
      model->Add<LayerNorm>(10);
      model->Add<Linear>(10, 2);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 16, false);
      model->Gradient(model->Parameters(), 0, gradient, 16);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  REQUIRE(CheckGradient(function) <= 1e-4);
}
*/

/**
 * Test that the functions that can access the parameters of the
 * Layer Norm layer work.
 *
TEST_CASE("LayerNormLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : size, eps.
  LayerNorm layer(5, 1e-3);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.InSize() == 5);
  REQUIRE(layer.Epsilon() == 1e-3);
}
*/

// /**
//  * Test if the AddMerge layer is able to forward the
//  * Forward/Backward/Gradient calls.
//  */
// TEST_CASE("AddMergeRunTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta, error;

//   AddMerge<> module(true, true);

//   Linear<>* linear = new Linear<>(10, 10);
//   module.Add(linear);

//   linear->Parameters().randu();
//   linear->Reset();

//   input = arma::zeros(10, 1);
//   module.Forward(input, output);

//   double parameterSum = arma::accu(linear->Parameters().submat(
//       100, 0, linear->Parameters().n_elem - 1, 0));

//   // Test the Backward function.
//   module.Backward(input, input, delta);

//   // Clean up before we break,
//   delete linear;

//   REQUIRE(parameterSum == Approx(arma::accu(output)).epsilon(1e-5));
//   REQUIRE(arma::accu(delta) == 0);
// }

/**
 * Test if the MultiplyMerge layer is able to forward the
 * Forward/Backward/Gradient calls.
 *
TEST_CASE("MultiplyMergeRunTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta, error;

  MultiplyMerge module(true, true);

  Linear* linear = new Linear(10, 10);
  module.Add(linear);

  linear->Parameters().randu();
  linear->Reset();

  input = arma::zeros(10, 1);
  module.Forward(input, output);

  double parameterSum = arma::accu(linear->Parameters().submat(
      100, 0, linear->Parameters().n_elem - 1, 0));

  // Test the Backward function.
  module.Backward(input, input, delta);

  // Clean up before we break,
  delete linear;

  REQUIRE(parameterSum == Approx(arma::accu(output)).epsilon(1e-5));
  REQUIRE(arma::accu(delta) == 0);
}
*/

/**
 * Simple subview module test.
 *
TEST_CASE("SimpleSubviewLayerTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta, outputMat;
  Subview moduleRow(1, 10, 19);

  // Test the Forward function for a vector.
  input = arma::ones(20, 1);
  moduleRow.Forward(input, output);
  REQUIRE(output.n_rows == 10);

  Subview moduleMat(4, 3, 6, 0, 2);

  // Test the Forward function for a matrix.
  input = arma::ones(20, 8);
  moduleMat.Forward(input, outputMat);
  REQUIRE(outputMat.n_rows == 12);
  REQUIRE(outputMat.n_cols == 2);

  // Test the Backward function.
  moduleMat.Backward(input, input, delta);
  REQUIRE(accu(delta) == 160);
  REQUIRE(delta.n_rows == 20);
}
*/

/**
 * Subview index test.
 *
TEST_CASE("SubviewIndexTest", "[ANNLayerTest]")
{
  arma::mat outputEnd, outputMid, outputStart, input, delta;
  input = arma::linspace<arma::vec>(1, 20, 20);

  // Slicing from the initial indices.
  Subview moduleStart(1, 0, 9);
  arma::mat subStart = arma::linspace<arma::vec>(1, 10, 10);

  moduleStart.Forward(input, outputStart);
  CheckMatrices(outputStart, subStart);

  // Slicing from the mid indices.
  Subview moduleMid(1, 6, 15);
  arma::mat subMid = arma::linspace<arma::vec>(7, 16, 10);

  moduleMid.Forward(input, outputMid);
  CheckMatrices(outputMid, subMid);

  // Slicing from the end indices.
  Subview moduleEnd(1, 10, 19);
  arma::mat subEnd = arma::linspace<arma::vec>(11, 20, 10);

  moduleEnd.Forward(input, outputEnd);
  CheckMatrices(outputEnd, subEnd);
}
*/

/**
 * Subview batch test.
 *
TEST_CASE("SubviewBatchTest", "[ANNLayerTest]")
{
  arma::mat output, input, outputCol, outputMat, outputDef;

  // All rows selected.
  Subview moduleCol(1, 0, 19);

  // Test with inSize 1.
  input = arma::ones(20, 8);
  moduleCol.Forward(input, outputCol);
  CheckMatrices(outputCol, input);

  // Few rows and columns selected.
  Subview moduleMat(4, 3, 6, 0, 2);

  // Test with inSize greater than 1.
  moduleMat.Forward(input, outputMat);
  output = arma::ones(12, 2);
  CheckMatrices(outputMat, output);

  // endCol changed to 3 by default.
  Subview moduleDef(4, 1, 6, 0, 4);

  // Test with inSize greater than 1 and endCol >= inSize.
  moduleDef.Forward(input, outputDef);
  output = arma::ones(24, 2);
  CheckMatrices(outputDef, output);
}
*/

/**
 * Test that the functions that can modify and access the parameters of the
 * Subview layer work.
 *
TEST_CASE("SubviewLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : inSize, beginRow, endRow, beginCol, endCol.
  Subview layer1(1, 2, 3, 4, 5);
  Subview layer2(1, 3, 4, 5, 6);

  // Make sure we can get the parameters correctly.
  REQUIRE(layer1.InSize() == 1);
  REQUIRE(layer1.BeginRow() == 2);
  REQUIRE(layer1.EndRow() == 3);
  REQUIRE(layer1.BeginCol() == 4);
  REQUIRE(layer1.EndCol() == 5);

  // Now modify the parameters to match the second layer.
  layer1.BeginRow() = 3;
  layer1.EndRow() = 4;
  layer1.BeginCol() = 5;
  layer1.EndCol() = 6;

  // Now ensure all results are the same.
  REQUIRE(layer1.InSize() == layer2.InSize());
  REQUIRE(layer1.BeginRow() == layer2.BeginRow());
  REQUIRE(layer1.EndRow() == layer2.EndRow());
  REQUIRE(layer1.BeginCol() == layer2.BeginCol());
  REQUIRE(layer1.EndCol() == layer2.EndCol());
}
*/

/*
 * Simple Reparametrization module test.
 *
TEST_CASE("SimpleReparametrizationLayerTest", "[ANNLayerTest]")
{
  arma::mat input, output, delta;
  Reparametrization module(5);

  // Test the Forward function. As the mean is zero and the standard
  // deviation is small, after multiplying the gaussian sample, the
  // output should be small enough.
  input = join_cols(arma::ones<arma::mat>(5, 1) * -15,
      arma::zeros<arma::mat>(5, 1));
  module.Forward(input, output);
  REQUIRE(arma::accu(output) <= 1e-5);

  // Test the Backward function.
  arma::mat gy = arma::zeros<arma::mat>(5, 1);
  module.Backward(input, gy, delta);
  REQUIRE(arma::accu(delta) != 0); // klBackward will be added.
}
*/

/**
 * Reparametrization module stochastic boolean test.
 *
TEST_CASE("ReparametrizationLayerStochasticTest", "[ANNLayerTest]")
{
  arma::mat input, outputA, outputB;
  Reparametrization module(5, false);

  input = join_cols(arma::ones<arma::mat>(5, 1),
      arma::zeros<arma::mat>(5, 1));

  // Test if two forward passes generate same output.
  module.Forward(input, outputA);
  module.Forward(input, outputB);

  CheckMatrices(outputA, outputB);
}
*/

/**
 * Reparametrization module includeKl boolean test.
 *
TEST_CASE("ReparametrizationLayerIncludeKlTest", "[ANNLayerTest]")
{
  arma::mat input, output, gy, delta;
  Reparametrization module(5, true, false);

  input = join_cols(arma::ones<arma::mat>(5, 1),
      arma::zeros<arma::mat>(5, 1));
  module.Forward(input, output);

  // As KL divergence is not included, with the above inputs, the delta
  // matrix should be all zeros.
  gy = arma::zeros(output.n_rows, output.n_cols);
  module.Backward(output, gy, delta);

  REQUIRE(arma::accu(delta) == 0);
}
*/

/**
 * Jacobian Reparametrization module test.
 *
TEST_CASE("JacobianReparametrizationLayerTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t inputElementsHalf = RandInt(2, 10);

    arma::mat input;
    input.set_size(inputElementsHalf * 2, 1);

    Reparametrization module(inputElementsHalf, false, false);

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Reparametrization layer numerical gradient test.
 *
TEST_CASE("GradientReparametrizationLayerTest", "[ANNLayerTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 1)),
        target(arma::mat("0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<IdentityLayer>();
      model->Add<Linear>(10, 6);
      model->Add<Reparametrization>(3, false, true, 1);
      model->Add<Linear>(3, 2);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  // REQUIRE(CheckGradient(function) <= 1e-4);
}
*/

/**
 * Reparametrization layer beta numerical gradient test.
 *
TEST_CASE("GradientReparametrizationLayerBetaTest", "[ANNLayerTest]")
{
  // Linear function gradient instantiation.
  struct GradientFunction
  {
    GradientFunction() :
        input(arma::randu(10, 2)),
        target(arma::mat("0 0"))
    {
      model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
      model->ResetData(input, target);
      model->Add<IdentityLayer>();
      model->Add<Linear>(10, 6);
      // Use a value of beta not equal to 1.
      model->Add<Reparametrization>(3, false, true, 2);
      model->Add<Linear>(3, 2);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, 1);
      model->Gradient(model->Parameters(), 0, gradient, 1);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
    arma::mat input, target;
  } function;

  // REQUIRE(CheckGradient(function) <= 1e-4);
}
*/

/**
 * Test that the functions that can access the parameters of the
 * Reparametrization layer work.
 *
TEST_CASE("ReparametrizationLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : latentSize, stochastic, includeKL, beta.
  Reparametrization layer(5, false, false, 2);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.OutputSize() == 5);
  REQUIRE(layer.Stochastic() == false);
  REQUIRE(layer.IncludeKL() == false);
  REQUIRE(layer.Beta() == 2);
}
*/

/**
 * Simple residual module test.
 *
TEST_CASE("SimpleResidualLayerTest", "[ANNLayerTest]")
{
  arma::mat outputA, outputB, input, deltaA, deltaB;

  Sequential* sequential = new Sequential(true);
  Residual* residual = new Residual(true);

  Linear* linearA = new Linear(10, 10);
  linearA->Parameters().randu();
  linearA->Reset();
  Linear* linearB = new Linear(10, 10);
  linearB->Parameters().randu();
  linearB->Reset();

  // Add the same layers (with the same parameters) to both Sequential and
  // Residual object.
  sequential->Add(linearA);
  sequential->Add(linearB);

  residual->Add(linearA);
  residual->Add(linearB);

  // Test the Forward function (pass the same input to both).
  input = arma::randu(10, 1);
  sequential->Forward(input, outputA);
  residual->Forward(input, outputB);

  CheckMatrices(outputA, outputB - input);

  // Test the Backward function (pass the same error to both).
  sequential->Backward(input, input, deltaA);
  residual->Backward(input, input, deltaB);

  CheckMatrices(deltaA, deltaB - input);

  delete sequential;
  delete residual;
  delete linearA;
  delete linearB;
}
*/

/**
 * Simple Highway module test.
 *
TEST_CASE("SimpleHighwayLayerTest", "[ANNLayerTest]")
{
  arma::mat outputA, outputB, input, deltaA, deltaB;
  Sequential* sequential = new Sequential(true);
  Highway* highway = new Highway(10, true);
  highway->Parameters().zeros();
  highway->Reset();

  Linear* linearA = new Linear(10, 10);
  linearA->Parameters().randu();
  linearA->Reset();
  Linear* linearB = new Linear(10, 10);
  linearB->Parameters().randu();
  linearB->Reset();

  // Add the same layers (with the same parameters) to both Sequential and
  // Highway object.
  highway->Add(linearA);
  highway->Add(linearB);
  sequential->Add(linearA);
  sequential->Add(linearB);

  // Test the Forward function (pass the same input to both).
  input = arma::randu(10, 1);
  sequential->Forward(input, outputA);
  highway->Forward(input, outputB);

  CheckMatrices(outputB, input * 0.5 + outputA * 0.5);

  delete sequential;
  delete highway;
  delete linearA;
  delete linearB;
}
*/

/**
 * Test that the function that can access the inSize parameter of the
 * Highway layer works.
 *
TEST_CASE("HighwayLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : inSize, model.
  Highway layer(1, true);

  // Make sure we can get the parameter successfully.
  REQUIRE(layer.InSize() == 1);
}
*/

// /**
//  * Sequential layer numerical gradient test.
//  */
// TEST_CASE("GradientHighwayLayerTest", "[ANNLayerTest]")
// {
//   // Linear function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(5, 1)),
//         target(arma::mat("0"))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer>();
//       model->Add<Linear>(5, 10);

//       highway = new Highway(10);
//       highway->Add<Linear>(10, 10);
//       highway->Add<ReLULayer>();
//       highway->Add<Linear>(10, 10);
//       highway->Add<ReLULayer>();

//       model->Add(highway);
//       model->Add<Linear>(10, 2);
//       model->Add<LogSoftMax>();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     Highway* highway;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

/**
 * Sequential layer numerical gradient test.
 */
// TEST_CASE("GradientSequentialLayerTest", "[ANNLayerTest]")
// {
//   // Linear function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(10, 1)),
//         target(arma::mat("0"))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<IdentityLayer>();
//       model->Add<Linear>(10, 10);
//       sequential = new Sequential();
//       sequential->Add<Linear>(10, 10);
//       sequential->Add<ReLULayer>();
//       sequential->Add<Linear>(10, 5);
//       sequential->Add<ReLULayer>();

//       model->Add(sequential);
//       model->Add<Linear>(5, 2);
//       model->Add<LogSoftMax>();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     Sequential* sequential;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

// /**
//  * WeightNorm layer numerical gradient test.
//  */
// TEST_CASE("GradientWeightNormLayerTest", "[ANNLayerTest]")
// {
//   // Linear function gradient instantiation.
//   struct GradientFunction
//   {
//     GradientFunction() :
//         input(arma::randu(10, 1)),
//         target(arma::mat("0"))
//     {
//       model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
//       model->ResetData(input, target);
//       model->Add<Linear>(10, 10);

//       Linear* linear = new Linear(10, 2);
//       weightNorm = new WeightNorm(linear);

//       model->Add(weightNorm);
//       model->Add<LogSoftMax>();
//     }

//     ~GradientFunction()
//     {
//       delete model;
//     }

//     double Gradient(arma::mat& gradient) const
//     {
//       double error = model->Evaluate(model->Parameters(), 0, 1);
//       model->Gradient(model->Parameters(), 0, gradient, 1);
//       return error;
//     }

//     arma::mat& Parameters() { return model->Parameters(); }

//     FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
//     WeightNorm* weightNorm;
//     arma::mat input, target;
//   } function;

//   REQUIRE(CheckGradient(function) <= 1e-4);
// }

// /**
//  * Test if the WeightNorm layer is able to forward the
//  * Forward/Backward/Gradient calls.
//  */
// TEST_CASE("WeightNormRunTest", "[ANNLayerTest]")
// {
//   arma::mat output, input, delta, error;
//   Linear* linear = new Linear(10, 10);

//   WeightNorm module(linear);

//   module.Parameters().randu();
//   module.Reset();

//   linear->Bias().zeros();

//   input = arma::zeros(10, 1);
//   module.Forward(input, output);

//   // Test the Backward function.
//   module.Backward(input, input, delta);

//   REQUIRE(0 == arma::accu(output));
//   REQUIRE(arma::accu(delta) == 0);
// }

// /**
//  * Simple serialization test for layer normalization layer.
//  */
// TEST_CASE("LayerNormSerializationTest", "[ANNLayerTest]")
// {
//   LayerNorm<> layer(10);
//   ANNLayerSerializationTest(layer);
// }

/**
 * Test that the padding options in Transposed Convolution layer.
 *
TEST_CASE("TransposedConvolutionLayerPaddingTest", "[ANNLayerTest]")
{
  arma::mat output, input, delta;

  TransposedConvolution module1(1, 1, 3, 3, 1, 1, 0, 0, 4, 4, 6, 6, "VALID");
  // Test the forward function.
  // Valid Should give the same result.
  input = arma::linspace<arma::colvec>(0, 15, 16);
  module1.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module1.Reset();
  module1.Forward(input, output);
  // Value calculated using tensorflow.nn.conv2d_transpose().
  REQUIRE(arma::accu(output) == 0.0);

  // Test the Backward Function.
  module1.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0.0);

  // Test Valid for non zero padding.
  TransposedConvolution module2(1, 1, 3, 3, 2, 2,
      std::tuple<size_t, size_t>(0, 0), std::tuple<size_t, size_t>(0, 0),
      2, 2, 5, 5, "VALID");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module2.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module2.Parameters()(2) = 8.0;
  module2.Parameters()(4) = 6.0;
  module2.Parameters()(6) = 4.0;
  module2.Parameters()(8) = 2.0;
  module2.Reset();
  module2.Forward(input, output);
  // Value calculated using torch.nn.functional.conv_transpose2d().
  REQUIRE(arma::accu(output) == 120.0);

  // Test the Backward Function.
  module2.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 960.0);

  // Test for same padding type.
  TransposedConvolution module3(1, 1, 3, 3, 2, 2, 0, 0, 3, 3, 3, 3, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 8, 9);
  module3.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module3.Reset();
  module3.Forward(input, output);
  REQUIRE(arma::accu(output) == 0);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test the Backward Function.
  module3.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0.0);

  // Output shape should equal input.
  TransposedConvolution module4(1, 1, 3, 3, 1, 1,
    std::tuple<size_t, size_t>(2, 2), std::tuple<size_t, size_t>(2, 2),
    5, 5, 5, 5, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module4.Parameters() = arma::mat(9 + 1, 1, arma::fill::zeros);
  module4.Reset();
  module4.Forward(input, output);
  REQUIRE(arma::accu(output) == 0);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test the Backward Function.
  module4.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0.0);

  TransposedConvolution module5(1, 1, 3, 3, 2, 2, 0, 0, 2, 2, 2, 2, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 3, 4);
  module5.Parameters() = arma::mat(25 + 1, 1, arma::fill::zeros);
  module5.Reset();
  module5.Forward(input, output);
  REQUIRE(arma::accu(output) == 0);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test the Backward Function.
  module5.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0.0);

  TransposedConvolution module6(1, 1, 4, 4, 1, 1, 1, 1, 5, 5, 5, 5, "SAME");
  // Test the forward function.
  input = arma::linspace<arma::colvec>(0, 24, 25);
  module6.Parameters() = arma::mat(16 + 1, 1, arma::fill::zeros);
  module6.Reset();
  module6.Forward(input, output);
  REQUIRE(arma::accu(output) == 0);
  REQUIRE(output.n_rows == input.n_rows);
  REQUIRE(output.n_cols == input.n_cols);

  // Test the Backward Function.
  module6.Backward(input, output, delta);
  REQUIRE(arma::accu(delta) == 0.0);
}
*/

/**
 * Simple test for Lp Pooling layer.
 */
// TEST_CASE("LpMaxPoolingTestCase", "[ANNLayerTest]")
// {
//   // For rectangular input to pooling layers.
//   arma::mat input = arma::mat(8, 1);
//   arma::mat output;
//   input.zeros();
//   input(0) = input(6) = 30;
//   input(1) = input(7) = 120;
//   input(2) = input(4) = 272;
//   input(3) = input(5) = 315;
//   // Output-Size should be 1 x 2.
//   // Square output.
//   LpPooling<> module1(4, 2, 2, 2, 2);
//   module1.InputHeight() = 2;
//   module1.InputWidth() = 4;
//   module1.Forward(input, output);
//   // Calculated using torch.nn.LPPool2d().
//   REQUIRE(arma::accu(output) - 706.0 == Approx(0.0).margin(2e-5));
//   REQUIRE(output.n_elem == 2);
//
//   // For Square input.
//   input = arma::mat(16, 1);
//   input.zeros();
//   input(0) = 4;
//   input(1) = 3;
//   input(3) = 12;
//   input(7) = 35;
//   input(8) = 6;
//   input(11) = 7;
//   input(12) = 8;
//   input(15) = 24;
//   // Output-Size should be 2 x 2.
//   // Square output.
//   LpPooling<> module3(2, 2, 2, 2, 2);
//   module3.InputHeight() = 4;
//   module3.InputWidth() = 4;
//   module3.Forward(input, output);
//   // Calculated using torch.nn.LPPool2d().
//   REQUIRE(arma::accu(output) - 77.0 == Approx(0.0).margin(2e-5));
//   REQUIRE(output.n_elem == 4);
// }

/**
 * Simple test for AddMerge layer.
 */
TEST_CASE("AddMergeTestCase", "[ANNLayerTest]")
{
  // For rectangular input to pooling layers.
  arma::mat input = arma::mat(28, 1);
  input.zeros();
  input(0) = input(16) = 1;
  input(1) = input(17) = 2;
  input(2) = input(18) = 3;
  input(3) = input(19) = 4;
  input(4) = input(20) = 5;
  input(5) = input(23) = 6;
  input(6) = input(24) = 7;
  input(14) = input(25) = 8;
  input(15) = input(26) = 9;

  AddMerge module1;
  module1.Add<MeanPooling>(2, 2, 2, 2, false);
  module1.Add<MeanPooling>(2, 2, 2, 2, false);

  AddMerge module2;
  module2.Add<MeanPooling>(2, 2, 2, 2, true);
  module2.Add<MeanPooling>(2, 2, 2, 2, true);

  module1.InputDimensions() = std::vector<size_t>({ 7, 4 });
  module1.ComputeOutputDimensions();
  module2.InputDimensions() = std::vector<size_t>({ 7, 4 });
  module2.ComputeOutputDimensions();

  // Calculated using torch.nn.MeanPool2d().
  arma::mat result1, result2;
  result1  <<  1.5000  <<  8.5000  <<  arma::endr
           <<  3.5000  <<  8.0000  <<  arma::endr
           <<  5.5000  <<  12.0000 <<  arma::endr
           <<  7.0000  <<  5.0000  <<  arma::endr;

  result2  <<  1.5000  <<  8.5000  <<  arma::endr
           <<  3.5000  <<  8.0000  <<  arma::endr
           <<  5.5000  <<  12.0000 <<  arma::endr;

  arma::mat output1, output2;
  output1.set_size(8, 1);
  output2.set_size(6, 1);
  module1.Forward(input, output1);
  REQUIRE(arma::accu(output1) == 51.0);
  module2.Forward(input, output2);
  REQUIRE(arma::accu(output2) == 39.0);
  output1.reshape(4, 2);
  output2.reshape(3, 2);
  CheckMatrices(output1, result1, 1e-1);
  CheckMatrices(output2, result2, 1e-1);

  arma::mat prevDelta1, prevDelta2;
  prevDelta1  << 3.6000 << -0.9000 << arma::endr
              << 3.6000 << -0.9000 << arma::endr
              << 3.6000 << -0.9000 << arma::endr
              << 3.6000 << -0.9000 << arma::endr;

  prevDelta2  << 3.6000 << -0.9000 << arma::endr
              << 3.6000 << -0.9000 << arma::endr
              << 3.6000 << -0.9000 << arma::endr;
  arma::mat delta1, delta2;
  delta1.set_size(28, 1);
  delta2.set_size(28, 1);
  prevDelta1.reshape(8, 1);
  prevDelta2.reshape(6, 1);
  module1.Backward(input, prevDelta1, delta1);
  REQUIRE(arma::accu(delta1) == Approx(21.6).epsilon(1e-3));
  module2.Backward(input, prevDelta2, delta2);
  REQUIRE(arma::accu(delta2) == Approx(16.2).epsilon(1e-3));
}

/**
 * Complex test for AddMerge layer.
 * This test includes: 
 * 1. AddMerge layer inside the AddMerge layer.
 * 2. Batch Size > 1.
 * 3. AddMerge layer with single child layer.
 */
TEST_CASE("AddMergeAdvanceTestCase", "[ANNLayerTest]")
{
  AddMerge r;
  AddMerge* r2 = new AddMerge();
  r2->Add<Linear>(5);
  r.Add<Linear>(5);
  r.Add(r2);
  r.InputDimensions() = std::vector<size_t>({ 5 });
  r.ComputeOutputDimensions();
  arma::mat rParams(r.WeightSize(), 1);
  r.SetWeights((double*) rParams.memptr());
  r.Network()[0]->Parameters().fill(2.0);
  ((AddMerge*) r.Network()[1])->Network()[0]->Parameters().fill(-1.0);

  Linear l(5);
  l.InputDimensions() = std::vector<size_t>({ 5 });
  l.ComputeOutputDimensions();
  arma::mat lParams(l.WeightSize(), 1);
  l.SetWeights((double*) lParams.memptr());
  l.Parameters().fill(1.0);

  arma::mat input(arma::randn(5, 10));
  arma::mat output1, output2;
  output1.set_size(5, 10);
  output2.set_size(5, 10);

  r.Forward(input, output1);
  l.Forward(input, output2);

  CheckMatrices(output1, output2, 1e-3);

  arma::mat delta1, delta2;
  delta1.set_size(5, 10);
  delta2.set_size(5, 10);
  r.Backward(input, output1, delta1);
  l.Backward(input, output2, delta2);

  CheckMatrices(output1, output2, 1e-3);
}

/**
 * Simple test for Identity layer.
 */
TEST_CASE("IdentityTestCase", "[ANNLayerTest]")
{
  // For rectangular input to pooling layers.
  arma::mat input = arma::mat(12, 1, arma::fill::randn);
  arma::mat output;
  // Output-Size should be 4 x 3.
  output.set_size(12, 1);

  Identity module1;
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  module1.Forward(input, output);
  CheckMatrices(output, input, 1e-1);
  REQUIRE(output.n_elem == 12);
  REQUIRE(output.n_cols == 1);
  REQUIRE(input.memptr() != output.memptr());

  arma::mat prevDelta = arma::mat(12, 1, arma::fill::randn);
  arma::mat delta;
  delta.set_size(12, 1);
  module1.Backward(input, prevDelta, delta);
  CheckMatrices(delta, prevDelta, 1e-1);
  REQUIRE(delta.memptr() != prevDelta.memptr());
}

/**
 * Test that the functions that can modify and access the parameters of the
 * Glimpse layer work.
 *
TEST_CASE("GlimpseLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : inSize, size, depth, scale, inputWidth, inputHeight.
  Glimpse layer1(1, 2, 3, 4, 5, 6);
  Glimpse layer2(1, 2, 3, 4, 6, 7);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer1.InputHeight() == 6);
  REQUIRE(layer1.InputWidth() == 5);
  REQUIRE(layer1.Scale() == 4);
  REQUIRE(layer1.Depth() == 3);
  REQUIRE(layer1.GlimpseSize() == 2);
  REQUIRE(layer1.InSize() == 1);

  // Now modify the parameters to match the second layer.
  layer1.InputHeight() = 7;
  layer1.InputWidth() = 6;

  // Now ensure that all the results are the same.
  REQUIRE(layer1.InputHeight() == layer2.InputHeight());
  REQUIRE(layer1.InputWidth() == layer2.InputWidth());
  REQUIRE(layer1.Scale() == layer2.Scale());
  REQUIRE(layer1.Depth() == layer2.Depth());
  REQUIRE(layer1.GlimpseSize() == layer2.GlimpseSize());
  REQUIRE(layer1.InSize() == layer2.InSize());
}
*/

/**
 * Test that the function that can access the stdev parameter of the
 * Reinforce Normal layer works.
 *
TEST_CASE("ReinforceNormalLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter : stdev.
  ReinforceNormal layer(4.0);

  // Make sure we can get the parameter successfully.
  REQUIRE(layer.StandardDeviation() == 4.0);
}
*/

/*
TEST_CASE("TransposedConvolutionalLayerOptionalParameterTest", "[ANNLayerTest]")
{
  Sequential* decoder = new Sequential();

  // Check if we can create an object without specifying output.
  REQUIRE_NOTHROW(decoder->Add<TransposedConvolution>(24, 16,
      5, 5, 1, 1, 0, 0, 10, 10));

  REQUIRE_NOTHROW(decoder->Add<TransposedConvolution>(16, 1,
      15, 15, 1, 1, 1, 1, 14, 14));

  delete decoder;
}
*/

// /**
//  * Linear module weight initialization test.
//  */
// TEST_CASE("LinearLayerWeightInitializationTest", "[ANNLayerTest]")
// {
//   size_t inSize = 10, outSize = 4;
//   Linear<> linear = Linear<>(inSize, outSize);
//   linear.Reset();
//   RandomInitialization().Initialize(linear.Weight());
//   linear.Bias().ones();

//   REQUIRE(std::equal(linear.Weight().begin(),
//       linear.Weight().end(), linear.Parameters().begin()));

//   REQUIRE(std::equal(linear.Bias().begin(),
//       linear.Bias().end(), linear.Parameters().begin() + inSize * outSize));

//   REQUIRE(linear.Weight().n_rows == outSize);
//   REQUIRE(linear.Weight().n_cols == inSize);
//   REQUIRE(linear.Bias().n_rows == outSize);
//   REQUIRE(linear.Bias().n_cols == 1);
//   REQUIRE(linear.Parameters().n_rows == inSize * outSize + outSize);
// }

// /**
//  * Atrous Convolution module weight initialization test.
//  */
// TEST_CASE("AtrousConvolutionLayerWeightInitializationTest", "[ANNLayerTest]")
// {
//   size_t inSize = 2, outSize = 3;
//   size_t kernelWidth = 4, kernelHeight = 5;
//   AtrousConvolution<> module = AtrousConvolution<>(inSize, outSize,
//       kernelWidth, kernelHeight, 6, 7, std::make_tuple(8, 9),
//       std::make_tuple(10, 11), 12, 13, 14, 15);
//   module.Reset();
//   RandomInitialization().Initialize(module.Weight());
//   module.Bias().ones();

//   REQUIRE(std::equal(module.Weight().begin(),
//       module.Weight().end(), module.Parameters().begin()));

//   REQUIRE(std::equal(module.Bias().begin(),
//       module.Bias().end(), module.Parameters().end() - outSize));

//   REQUIRE(module.Weight().n_rows == kernelWidth);
//   REQUIRE(module.Weight().n_cols == kernelHeight);
//   REQUIRE(module.Weight().n_slices == inSize * outSize);
//   REQUIRE(module.Bias().n_rows == outSize);
//   REQUIRE(module.Bias().n_cols == 1);
//   REQUIRE(module.Parameters().n_rows
//       == (outSize * inSize * kernelWidth * kernelHeight) + outSize);
// }

/**
 * Transposed Convolution module weight initialization test.
 *
TEST_CASE("TransposedConvolutionWeightInitializationTest", "[ANNLayerTest]")
{
  size_t inSize = 3, outSize = 3;
  size_t kernelWidth = 4, kernelHeight = 4;
  TransposedConvolution module = TransposedConvolution(inSize, outSize,
      kernelWidth, kernelHeight, 1, 1, 1, 1, 5, 5, 6, 6);
  module.Reset();
  RandomInitialization().Initialize(module.Weight());
  module.Bias().ones();

  REQUIRE(std::equal(module.Weight().begin(),
      module.Weight().end(), module.Parameters().begin()));

  REQUIRE(std::equal(module.Bias().begin(),
      module.Bias().end(), module.Parameters().end() - outSize));

  REQUIRE(module.Weight().n_rows == kernelWidth);
  REQUIRE(module.Weight().n_cols == kernelHeight);
  REQUIRE(module.Weight().n_slices == inSize * outSize);
  REQUIRE(module.Bias().n_rows == outSize);
  REQUIRE(module.Bias().n_cols == 1);
  REQUIRE(module.Parameters().n_rows
      == (outSize * inSize * kernelWidth * kernelHeight) + outSize);
}
*/

/**
 * Simple Test for ChannelShuffle layer.
 */
// TEST_CASE("ChannelShuffleLayerTest", "[ANNLayerTest]")
// {
//   arma::mat input1, output1, outputExpected1, outputBackward1;
//   ChannelShuffle<> module1(2, 2, 6, 2);
//
//   input1 << 1  << 13 << arma::endr
//          << 2  << 14 << arma::endr
//          << 3  << 15 << arma::endr
//          << 4  << 16 << arma::endr
//          << 5  << 17 << arma::endr
//          << 6  << 18 << arma::endr
//          << 7  << 19 << arma::endr
//          << 8  << 20 << arma::endr
//          << 9  << 21 << arma::endr
//          << 10 << 22 << arma::endr
//          << 11 << 23 << arma::endr
//          << 12 << 24 << arma::endr;
//   input1.reshape(24, 1);
//   // Value calculated using torch.nn.ChannelShuffle().
//   outputExpected1 << 1  << 17 << arma::endr
//                   << 2  << 18 << arma::endr
//                   << 3  << 19 << arma::endr
//                   << 4  << 20 << arma::endr
//                   << 13 << 9 << arma::endr
//                   << 14 << 10 << arma::endr
//                   << 15 << 11 << arma::endr
//                   << 16 << 12 << arma::endr
//                   << 5  << 21 << arma::endr
//                   << 6  << 22 << arma::endr
//                   << 7  << 23 << arma::endr
//                   << 8  << 24 << arma::endr;
//   outputExpected1.reshape(24, 1);
//   // Check the Forward pass of the layer.
//   module1.Forward(input1, output1);
//   CheckMatrices(output1, outputExpected1);
//
//   // Check the Backward pass of the layer.
//   module1.Backward(output1, output1, outputBackward1);
//   CheckMatrices(input1, outputBackward1);
//
// }

/**
 * Simple Test for PixelShuffle layer.
 */
// TEST_CASE("PixelShuffleLayerTest", "[ANNLayerTest]")
// {
//   arma::mat input1, output1, gy1, g1, outputExpected1, gExpected1;
//   arma::mat input2, output2, gy2, g2, outputExpected2, gExpected2;
//   PixelShuffle<> module1(2, 2, 2, 4);
//   PixelShuffle<> module2(2, 2, 2, 4);
//
//   // Input is a single image, of size (2,2) and having 4 channels.
//   input1 << 1 << 3 << 2 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0
//       << 0 << 0 << arma::endr;
//   gy1 << 1 << 5 << 9 << 13 << 2 << 6 << 10 << 14 << 3 << 7 << 11 << 15 << 4 << 8
//       << 12 << 16 << arma::endr;
//
//   // Calculated using torch.nn.PixelShuffle().
//   outputExpected1 << 1 << 0 << 3 << 0 << 0 << 0 << 0 << 0 << 2 << 0 << 4 << 0
//       << 0 << 0 << 0 << 0 << arma::endr;
//   gExpected1 << 1 << 9 << 3 << 11 << 5 << 13 << 7 << 15 << 2 << 10 << 4 << 12
//       << 6 << 14 << 8 << 16 << arma::endr;
//
//   input1 = input1.t();
//   outputExpected1 = outputExpected1.t();
//   gy1 = gy1.t();
//   gExpected1 = gExpected1.t();
//
//   // Check the Forward pass of the layer.
//   module1.Forward(input1, output1);
//   CheckMatrices(output1, outputExpected1);
//
//   // Check the Backward pass of the layer.
//   module1.Backward(input1, gy1, g1);
//   CheckMatrices(g1, gExpected1);
//
//   // Input is a batch of 2 images, each of size (2,2) and having 4 channels.
//   input2 << 1 << 3 << 2 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0
//       << 0 << 0 << arma::endr << 5 << 7 << 6 << 8 << 0 << 0 << 0 << 0 << 0 << 0
//       << 0 << 0 << 0 << 0 << 0 << 0 << arma::endr;
//   gy2 << 1 << 5 << 9 << 13 << 2 << 6 << 10 << 14 << 3 << 7 << 11 << 15 << 4 << 8
//       << 12 << 16 << arma::endr << 17 << 21 << 25 << 29 << 18 << 22 << 26 << 30
//       << 19 << 23 << 27 << 31 << 20 << 24 << 28 << 32 << arma::endr;
//
//   // Calculated using torch.nn.PixelShuffle().
//   outputExpected2 << 1 << 0 << 3 << 0 << 0 << 0 << 0 << 0 << 2 << 0 << 4 << 0
//       << 0 << 0 << 0 << 0 << arma::endr << 5 << 0 << 7 << 0 << 0 << 0 << 0 << 0
//       << 6 << 0 << 8 << 0 << 0 << 0 << 0 << 0 << arma::endr;
//   gExpected2 << 1 << 9 << 3 << 11 << 5 << 13 << 7 << 15 << 2 << 10 << 4 << 12
//       << 6 << 14 << 8 << 16 << arma::endr << 17 << 25 << 19 << 27 << 21 << 29
//       << 23 << 31 << 18 << 26 << 20 << 28 << 22 << 30 << 24 << 32 << arma::endr;
//
//   input2 = input2.t();
//   outputExpected2 = outputExpected2.t();
//   gy2 = gy2.t();
//   gExpected2 = gExpected2.t();
//
//   // Check the Forward pass of the layer.
//   module2.Forward(input2, output2);
//   CheckMatrices(output2, outputExpected2);
//
//   // Check the Backward pass of the layer.
//   module2.Backward(input2, gy2, g2);
//   CheckMatrices(g2, gExpected2);
// }

/**
 * Test that the function that can access the parameters of the
 * PixelShuffle layer works.
 */
// TEST_CASE("PixelShuffleLayerParametersTest", "[ANNLayerTest]")
// {
//   // Create the layer using the empty constructor.
//   PixelShuffle<> layer;
//
//   // Set the different input parameters of the layer.
//   layer.UpscaleFactor() = 2;
//   layer.InputHeight() = 2;
//   layer.InputWidth() = 2;
//   layer.InputChannels() = 4;
//
//   // Make sure we can get the parameters successfully.
//   REQUIRE(layer.UpscaleFactor() == 2);
//   REQUIRE(layer.InputHeight() == 2);
//   REQUIRE(layer.InputWidth() == 2);
//   REQUIRE(layer.InputChannels() == 4);
//
//   arma::mat input, output;
//   // Input is a batch of 2 images, each of size (2,2) and having 4 channels.
//   input << 1 << 3 << 2 << 4 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0 << 0
//       << 0 << 0 << arma::endr << 5 << 7 << 6 << 8 << 0 << 0 << 0 << 0 << 0 << 0
//       << 0 << 0 << 0 << 0 << 0 << 0 << arma::endr;
//   input = input.t();
//   layer.Forward(input, output);
//
//   // Check whether output parameters are returned correctly.
//   REQUIRE(layer.OutputHeight() == 4);
//   REQUIRE(layer.OutputWidth() == 4);
//   REQUIRE(layer.OutputChannels() == 1);
// }

// /**
//  * Simple Test for SpatialDropout layer.
//  */
// TEST_CASE("SpatialDropoutLayerTest", "[ANNLayerTest]")
// {
//   arma::mat input, output, gy, g, temp;
//   arma::mat outputsExpected = arma::zeros(8, 12);
//   arma::mat gsExpected = arma::zeros(8, 12);

//   // Set the seed to a random value.
//   arma::arma_rng::set_seed_random();
//   SpatialDropout<> module(3, 0.2);

//   // Input is a batch of 2 images, each of size (2,2) and having 4 channels.
//   input = { 0.4963, 0.0885, 0.7682, 0.1320, 0.3074, 0.4901, 0.6341, 0.8964,
//       0.4556, 0.3489, 0.6323, 0.4017 };
//
//   gy = { 1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12 };
//
//   // Following values have been calculated using torch.nn.Dropout2d(p=0.2).
//   temp = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//   outputsExpected.row(0) = temp;
//   temp = { 0, 0, 0, 0, 0.3842, 0.6126, 0.7926, 1.1205, 0.5695, 0.4361, 0.7904,
//       0.5021 };
//   outputsExpected.row(1) = temp;
//   temp = { 0.6204, 0.1106, 0.9603, 0.1650, 0, 0, 0, 0, 0.5695, 0.4361,
//       0.7904, 0.5021 };
//   outputsExpected.row(2) = temp;
//   temp = { 0.6204, 0.1106, 0.9603, 0.1650, 0.3842, 0.6126, 0.7926, 1.1205, 0,
//       0, 0, 0 };
//   outputsExpected.row(3) = temp;
//   temp = { 0, 0, 0, 0, 0, 0, 0, 0, 0.5695, 0.4361, 0.7904, 0.5021 };
//   outputsExpected.row(4) = temp;
//   temp = { 0, 0, 0, 0, 0.3842, 0.6126, 0.7926, 1.1205, 0, 0, 0, 0 };
//   outputsExpected.row(5) = temp;
//   temp = { 0.6204, 0.1106, 0.9603, 0.1650, 0, 0, 0, 0, 0, 0, 0, 0 };
//   outputsExpected.row(6) = temp;
//   temp = { 0.6204, 0.1106, 0.9603, 0.1650, 0.3842, 0.6126, 0.7926, 1.1205,
//       0.5695, 0.4361, 0.7904, 0.5021 };
//   outputsExpected.row(7) = temp;
//   temp = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//   gsExpected.row(0) = temp;
//   temp = { 0, 0, 0, 0, 6.2500, 8.7500, 7.5000, 10.0000, 11.2500, 13.7500,
//       12.5000, 15.0000 };
//   gsExpected.row(1) = temp;
//   temp = { 1.2500, 3.7500, 2.5000, 5.0000, 0, 0, 0, 0, 11.2500, 13.7500,
//       12.5000, 15.0000 };
//   gsExpected.row(2) = temp;
//   temp = { 1.2500, 3.7500, 2.5000, 5.0000, 6.2500, 8.7500, 7.5000, 10.0000, 0,
//       0, 0, 0 };
//   gsExpected.row(3) = temp;
//   temp = { 0, 0, 0, 0, 0, 0, 0, 0, 11.2500, 13.7500, 12.5000, 15.0000 };
//   gsExpected.row(4) = temp;
//   temp = { 0, 0, 0, 0, 6.2500, 8.7500, 7.5000, 10.0000, 0, 0, 0, 0 };
//   gsExpected.row(5) = temp;
//   temp = { 1.2500, 3.7500, 2.5000, 5.0000, 0, 0, 0, 0, 0, 0, 0, 0 };
//   gsExpected.row(6) = temp;
//   temp = { 1.2500, 3.7500, 2.5000, 5.0000, 6.2500, 8.7500, 7.5000, 10.0000,
//       11.2500, 13.7500, 12.5000, 15.0000 };
//   gsExpected.row(7) = temp;

//   input = input.t();
//   gy = gy.t();
//   outputsExpected = outputsExpected.t();
//   gsExpected = gsExpected.t();

//   // Compute the Forward and Backward passes and store the results.
//   module.Forward(input, output);
//   module.Backward(input, gy, g);

//   // Check through all possible cases, to find a match and then compare results.
//   for (size_t i = 0; i < outputsExpected.n_cols; ++i)
//   {
//     if (arma::approx_equal(outputsExpected.col(i), output, "absdiff", 1e-1))
//     {
//       // Check the correctness of the Forward pass of the layer.
//       CheckMatrices(output, outputsExpected.col(i), 1e-1);
//       // Check the correctness of the Backward pass of the layer.
//       CheckMatrices(g, gsExpected.col(i), 1e-1);
//     }
//   }

//   // Check if the output is same as input when using deterministic mode.
//   module.Deterministic() = true;
//   output.clear();
//   module.Forward(input, output);
//   CheckMatrices(output, input, 1e-1);
// }

// /**
//  * Test that the function that can access the parameters of the
//  * SpatialDropout layer works.
//  */
// TEST_CASE("SpatialDropoutLayerParametersTest", "[ANNLayerTest]")
// {
//   // Create the layer using the empty constructor.
//   SpatialDropout<> layer;

//   // Set the input parameters.
//   layer.Size() = 3;
//   layer.Ratio(0.2);

//   // Check whether the input parameters have been set correctly.
//   REQUIRE(layer.Size() == 3);
//   REQUIRE(layer.Ratio() == 0.2);
// }

/**
 * Simple Positional Encoding layer test.
 *
TEST_CASE("SimplePositionalEncodingTest", "[ANNLayerTest]")
{
  const size_t seqLength = 5;
  const size_t embedDim = 4;
  const size_t batchSize = 2;

  arma::mat input = arma::randu(embedDim * seqLength, batchSize);
  arma::mat gy = 0.01 * arma::randu(embedDim * seqLength, batchSize);
  arma::mat output, g;

  PositionalEncoding module(embedDim, seqLength);

  // Check Forward function.
  module.Forward(input, output);
  arma::mat pe = output - input;
  CheckMatrices(arma::mean(pe, 1), module.Encoding());

  // Check Backward function.
  module.Backward(input, gy, g);
  REQUIRE(std::equal(gy.begin(), gy.end(), g.begin()));
}
*/

/**
 * Jacobian test for Positional Encoding layer.
 *
TEST_CASE("JacobianPositionalEncodingTest", "[ANNLayerTest]")
{
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t embedDim = 4;
    const size_t seqLength = RandInt(5, 10);
    arma::mat input;
    input.set_size(embedDim * seqLength, 1);

    PositionalEncoding module(embedDim, seqLength);

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Simple Multihead Attention test.
 *
TEST_CASE("SimpleMultiheadAttentionTest", "[ANNLayerTest]")
{
  size_t tLen = 5;
  size_t sLen = tLen;
  size_t embedDim = 4;
  size_t numHeads = 2;
  size_t bsz = 3;

  arma::mat query = 0.1 * arma::randu(embedDim * tLen, bsz);
  arma::mat output;

  arma::mat attnMask = arma::zeros(tLen, sLen);
  for (size_t i = 0; i < tLen; ++i)
  {
    for (size_t j = 0; j < sLen; ++j)
    {
      if (i < j)
        attnMask(i, j) = std::numeric_limits<double>::lowest();
    }
  }

  arma::mat keyPaddingMask = arma::zeros(1, sLen);
  keyPaddingMask(sLen - 1) = std::numeric_limits<double>::lowest();

  MultiheadAttention module(tLen, sLen, embedDim, numHeads);
  module.AttentionMask() = attnMask;
  module.KeyPaddingMask() = keyPaddingMask;
  module.Reset();
  module.Parameters().randu();

  // Forward test.
  arma::mat input = arma::join_cols(arma::join_cols(query, query), query);

  module.Forward(input, output);
  REQUIRE(output.n_rows == embedDim * tLen);
  REQUIRE(output.n_cols == bsz);

  // Backward test.
  arma::mat gy = 0.01 * arma::randu(embedDim * tLen, bsz);
  arma::mat g;
  module.Backward(input, gy, g);
  REQUIRE(g.n_rows == input.n_rows);
  REQUIRE(g.n_cols == input.n_cols);

  // Gradient test.
  arma::mat error = 0.05 * arma::randu(embedDim * tLen, bsz);
  arma::mat gradient;
  module.Gradient(input, error, gradient);
  REQUIRE(gradient.n_rows == module.Parameters().n_rows);
  REQUIRE(gradient.n_cols == module.Parameters().n_cols);
}
*/

/**
 * Jacobian MultiheadAttention module test.
 *
TEST_CASE("JacobianMultiheadAttentionTest", "[ANNLayerTest]")
{
  // Check when query = key = value.
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t tgtSeqLen = 2;
    const size_t embedDim = 4;
    const size_t nHeads = 2;
    const size_t batchSize = 1;

    arma::mat query = arma::randu(embedDim * tgtSeqLen, batchSize);
    arma::mat input = arma::join_cols(arma::join_cols(query, query), query);

    MultiheadAttention module(tgtSeqLen, tgtSeqLen, embedDim, nHeads);
    module.Parameters().randu();

    double error = CustomJacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }

  // Check when key = value.
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t tgtSeqLen = 2;
    const size_t srcSeqLen = RandInt(2, 5);
    const size_t embedDim = 4;
    const size_t nHeads = 2;
    const size_t batchSize = 1;

    arma::mat query = arma::randu(embedDim * tgtSeqLen, batchSize);
    arma::mat key = 0.091 * arma::randu(embedDim * srcSeqLen, batchSize);
    arma::mat input = arma::join_cols(arma::join_cols(query, key), key);

    MultiheadAttention module(tgtSeqLen, srcSeqLen, embedDim, nHeads);
    module.Parameters().randu();

    double error = CustomJacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }

  // Check when query, key and value are not same.
  for (size_t i = 0; i < 5; ++i)
  {
    const size_t tgtSeqLen = 2;
    const size_t srcSeqLen = RandInt(2, 5);
    const size_t embedDim = 4;
    const size_t nHeads = 2;
    const size_t batchSize = 1;

    arma::mat query = arma::randu(embedDim * tgtSeqLen, batchSize);
    arma::mat key = 0.091 * arma::randu(embedDim * srcSeqLen, batchSize);
    arma::mat value = 0.045 * arma::randu(embedDim * srcSeqLen, batchSize);
    arma::mat input = arma::join_cols(arma::join_cols(query, key), value);

    MultiheadAttention module(tgtSeqLen, srcSeqLen, embedDim, nHeads);
    module.Parameters().randu();

    double error = JacobianTest(module, input);
    REQUIRE(error <= 1e-5);
  }
}
*/

/**
 * Numerical gradient test for MultiheadAttention layer.
 *
TEST_CASE("GradientMultiheadAttentionTest", "[ANNLayerTest]")
{
  struct GradientFunction
  {
    GradientFunction() :
        tgtSeqLen(2),
        srcSeqLen(2),
        embedDim(4),
        nHeads(2),
        vocabSize(5),
        batchSize(2)
    {
      input = arma::randu(embedDim * (tgtSeqLen + 2 * srcSeqLen), batchSize);
      target = arma::zeros(vocabSize, batchSize);
      for (size_t i = 0; i < target.n_elem; ++i)
      {
        const size_t label = RandInt(1, vocabSize);
        target(i) = label;
      }

      attnMask = arma::zeros(tgtSeqLen, srcSeqLen);
      for (size_t i = 0; i < tgtSeqLen; ++i)
      {
        for (size_t j = 0; j < srcSeqLen; ++j)
        {
          if (i < j)
            attnMask(i, j) = std::numeric_limits<double>::lowest();
        }
      }

      keyPaddingMask = arma::zeros(1, srcSeqLen);
      keyPaddingMask(srcSeqLen - 1) = std::numeric_limits<double>::lowest();

      model = new FFN<NegativeLogLikelihood, XavierInitialization>();
      model->ResetData(input, target);
      // attnModule = new MultiheadAttention(tgtSeqLen, srcSeqLen, embedDim,
      //     nHeads);
      // attnModule->AttentionMask() = attnMask;
      // attnModule->KeyPaddingMask() = keyPaddingMask;
      // model->Add(attnModule);
      model->Add<MultiheadAttention>(tgtSeqLen, srcSeqLen, embedDim, nHeads,
          attnMask, keyPaddingMask);
      model->Add<Linear>(embedDim * tgtSeqLen, vocabSize);
      model->Add<LogSoftMax>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(arma::mat& gradient) const
    {
      double error = model->Evaluate(model->Parameters(), 0, batchSize);
      model->Gradient(model->Parameters(), 0, gradient, batchSize);
      return error;
    }

    arma::mat& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihood, XavierInitialization>* model;
    // MultiheadAttention* attnModule;

    arma::mat input, target, attnMask, keyPaddingMask;
    const size_t tgtSeqLen;
    const size_t srcSeqLen;
    const size_t embedDim;
    const size_t nHeads;
    const size_t vocabSize;
    const size_t batchSize;
  } function;

  REQUIRE(CheckGradient(function) <= 3e-06);
}
*/

/**
 * Simple tests for instance normalization layer.
 *
TEST_CASE("InstanceNormLayerTest", "[ANNLayerTest]")
{
  arma::mat input, result, output, delta, deltaExpected;
  arma::mat runningMean, runningVar;

  // Represents 2 images, each having 3 channels, and shape (3,2).
  input << 1  << 19  << arma::endr
        << 2  << 20  << arma::endr
        << 3  << 21  << arma::endr
        << 4  << 22  << arma::endr
        << 5  << 23  << arma::endr
        << 6  << 24  << arma::endr
        << 7  << 25  << arma::endr
        << 8  << 26  << arma::endr
        << 9  << 27  << arma::endr
        << 10 << 28  << arma::endr
        << 11 << 29  << arma::endr
        << 12 << 30  << arma::endr
        << 13 << 31  << arma::endr
        << 14 << 32  << arma::endr
        << 15 << 33  << arma::endr
        << 16 << 34  << arma::endr
        << 17 << 35  << arma::endr
        << 18 << 36  << arma::endr;

  // Output calculated using torch.nn.InstanceNorm2d().
  result  << -1.4638  << -1.4638 << arma::endr
          << -0.8783  << -0.8783 << arma::endr
          << -0.2928  << -0.2928 << arma::endr
          <<  0.2928  <<  0.2928 << arma::endr
          <<  0.8783  <<  0.8783 << arma::endr
          <<  1.4638  <<  1.4638 << arma::endr
          << -1.4638  << -1.4638 << arma::endr
          << -0.8783  << -0.8783 << arma::endr
          << -0.2928  << -0.2928 << arma::endr
          <<  0.2928  <<  0.2928 << arma::endr
          <<  0.8783  <<  0.8783 << arma::endr
          <<  1.4638  <<  1.4638 << arma::endr
          << -1.4638  << -1.4638 << arma::endr
          << -0.8783  << -0.8783 << arma::endr
          << -0.2928  << -0.2928 << arma::endr
          <<  0.2928  <<  0.2928 << arma::endr
          <<  0.8783  <<  0.8783 << arma::endr
          <<  1.4638  <<  1.4638 << arma::endr;

  // Calculated using torch.nn.InstanceNorm2d().
  deltaExpected << 1.8367 <<  1.8367 << arma::endr
                << 0.3967 <<  0.3967 << arma::endr
                << 0.0147 <<  0.0147 << arma::endr
                <<-0.0147 << -0.0147 << arma::endr
                <<-0.3967 << -0.3967 << arma::endr
                <<-1.8367 << -1.8367 << arma::endr
                << 1.8367 <<  1.8367 << arma::endr
                << 0.3967 <<  0.3967 << arma::endr
                << 0.0147 <<  0.0147 << arma::endr
                <<-0.0147 << -0.0147 << arma::endr
                <<-0.3967 << -0.3967 << arma::endr
                <<-1.8367 << -1.8367 << arma::endr
                << 1.8367 <<  1.8367 << arma::endr
                << 0.3967 <<  0.3967 << arma::endr
                << 0.0147 <<  0.0147 << arma::endr
                <<-0.0147 << -0.0147 << arma::endr
                <<-0.3967 << -0.3967 << arma::endr
                <<-1.8367 << -1.8367 << arma::endr;

  // Check Forward and Backward pass in non-deterministic mode.
  InstanceNorm<> module(3, input.n_cols, 1e-5, false, 0.1);
  output.zeros(arma::size(input));
  module.Forward(input, output);
  CheckMatrices(output, result, 1e-1);

  module.Backward(input, output, delta);
  CheckMatrices(delta, deltaExpected, 1e-1);

  runningMean = arma::mat(3, 1);
  runningVar = arma::mat(3, 1);
  runningMean(0) = 1.2500;
  runningMean(1) = 1.8500;
  runningMean(2) = 2.4500;
  runningVar(0) = 1.2500;
  runningVar(1) = 1.2500;
  runningVar(2) = 1.2500;

  CheckMatrices(runningMean, module.TrainingMean(), 1e-1);
  CheckMatrices(runningVar, module.TrainingVariance(), 1e-1);

  // Check Forward pass in deterministic mode.
  InstanceNorm<> module1(3, input.n_cols, 1e-5, false, 0.1);
  module1.Deterministic() = true;
  output.zeros(arma::size(input));
  module1.Forward(input, output);

  // Calculated using torch.nn.InstanceNorm2d().
  result  <<  1.0000 <<  18.9999 << arma::endr
          <<  2.0000 <<  19.9999 << arma::endr
          <<  3.0000 <<  20.9999 << arma::endr
          <<  4.0000 <<  21.9999 << arma::endr
          <<  5.0000 <<  22.9999 << arma::endr
          <<  6.0000 <<  23.9999 << arma::endr
          <<  7.0000 <<  24.9999 << arma::endr
          <<  8.0000 <<  25.9999 << arma::endr
          <<  9.0000 <<  26.9999 << arma::endr
          << 10.0000 <<  27.9999 << arma::endr
          << 10.9999 <<  28.9999 << arma::endr
          << 11.9999 <<  29.9999 << arma::endr
          << 12.9999 <<  30.9998 << arma::endr
          << 13.9999 <<  31.9998 << arma::endr
          << 14.9999 <<  32.9998 << arma::endr
          << 15.9999 <<  33.9998 << arma::endr
          << 16.9999 <<  34.9998 << arma::endr
          << 17.9999 <<  35.9998 << arma::endr;

  CheckMatrices(output, result, 1e-1);
}
*/

/**
 * Test that the functions that can access the parameters of the
 * Instance Norm layer work.
 *
TEST_CASE("InstanceNormLayerParametersTest", "[ANNLayerTest]")
{
  // Parameter order : size, eps.
  InstanceNorm<> layer(7, 0, 1e-3);

  // Make sure we can get the parameters successfully.
  REQUIRE(layer.InputSize() == 7);
  REQUIRE(layer.Epsilon() == 1e-3);

  arma::mat runningMean(7, 1, arma::fill::randn);
  arma::mat runningVariance(7, 1, arma::fill::randn);

  layer.TrainingVariance() = runningVariance;
  layer.TrainingMean() = runningMean;
  CheckMatrices(layer.TrainingVariance(), runningVariance);
  CheckMatrices(layer.TrainingMean(), runningMean);
}
*/

/**
 * Instance Norm layer numerical gradient test.
 *
TEST_CASE("GradientInstanceNormLayerTest", "[ANNLayerTest]")
{
  // Add function gradient instantiation.
  // To make this test robust, check it ten times.
  bool pass = false;
  for (size_t trial = 0; trial < 10; trial++)
  {
    struct GradientFunction
    {
      GradientFunction()
      {
        input = arma::randn(16, 1024);
        arma::mat target;
        target.ones(1, 1024);

        model = new FFN<NegativeLogLikelihood, NguyenWidrowInitialization>();
        model->ResetData(input, target);
        model->Add<IdentityLayer<> >();
        model->Add<Convolution<> >(1, 2, 3, 3, 1, 1, 0, 0, 4, 4);
        model->Add<InstanceNorm<> > (2, 1024);
        model->Add<Linear<> >(2 * 2 * 2, 2);
        model->Add<LogSoftMax<> >();
      }

      ~GradientFunction()
      {
        delete model;
      }

      double Gradient(arma::mat& gradient) const
      {
        double error = model->Evaluate(model->Parameters(), 0, 1024, false);
        model->Gradient(model->Parameters(), 0, gradient, 1024);
        return error;
      }

      arma::mat& Parameters() { return model->Parameters(); }

      FFN<NegativeLogLikelihood, NguyenWidrowInitialization>* model;
      arma::mat input, target;
    } function;

    double gradient = CheckGradient(function);
    if (gradient < 1e-1)
    {
      pass = true;
      break;
    }
  }

  REQUIRE(pass);
}
*/
