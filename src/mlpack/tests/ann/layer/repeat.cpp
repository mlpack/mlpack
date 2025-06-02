/**
 * @file tests/ann/layer/repeat.cpp
 * @author Adam Kropp
 *
 * Tests the repeat layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann.hpp>

#include "../../test_catch_tools.hpp"
#include "../../catch.hpp"
#include "../../serialization.hpp"
#include "../ann_test_tools.hpp"

using namespace mlpack;

/**
 * Simple test for Repeat layer using interleaved repeats along axis 0.
 */
TEMPLATE_TEST_CASE("RepeatTestCaseI0", "[ANNLayerTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  // Input will be 4 x 3.
  MatType input(4, 3, arma::fill::randn);
  MatType target = arma::repelem(input, 2, 1);
  input.reshape(12, 1);
  target.reshape(24, 1);

  MatType output;
  // Output-Size should be 8 x 3.
  output.set_size(24, 1);

  RepeatType<MatType> module1({2, 1}, true);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 8);
  REQUIRE(module1.OutputDimensions()[1] == 3);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 24);
  REQUIRE(output.n_cols == 1);

  MatType prevDelta(24, 1, arma::fill::randn);
  MatType delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(8, 3);
  MatType targetDelta(4, 3);
  for (size_t i = 0; i < targetDelta.n_rows; i++)
  {
    targetDelta.row(i) = arma::sum(prevDelta.rows(i * 2, i * 2 + 1), 0) / 2;
  }
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Repeat layer using interleaved repeats along axis 1.
 */
TEMPLATE_TEST_CASE("RepeatTestCaseI1", "[ANNLayerTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  // Input will be 4 x 3.
  MatType input(4, 3, arma::fill::randn);
  MatType target = arma::repelem(input, 1, 2);
  input.reshape(12, 1);
  target.reshape(24, 1);

  MatType output;
  // Output-Size should be 4 x 6.
  output.set_size(24, 1);

  RepeatType<MatType> module1({1, 2}, true);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 4);
  REQUIRE(module1.OutputDimensions()[1] == 6);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 24);
  REQUIRE(output.n_cols == 1);

  MatType prevDelta(24, 1, arma::fill::randn);
  MatType delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(4, 6);
  MatType targetDelta(4, 3);
  for (size_t i = 0; i < targetDelta.n_cols; i++)
  {
    targetDelta.col(i) = arma::sum(prevDelta.cols(i * 2, i * 2 + 1), 1) / 2;
  }
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Repeat layer using interleaved repeats along axis 0 and 1.
 */
TEMPLATE_TEST_CASE("RepeatTestCaseI2", "[ANNLayerTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  // Input will be 4 x 3.
  MatType input(4, 3, arma::fill::randn);
  MatType target = arma::repelem(input, 2, 2);
  input.reshape(12, 1);
  target.reshape(48, 1);

  MatType output;
  // Output-Size should be 8 x 6.
  output.set_size(48, 1);

  RepeatType<MatType> module1({2, 2}, true);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 8);
  REQUIRE(module1.OutputDimensions()[1] == 6);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 48);
  REQUIRE(output.n_cols == 1);

  MatType prevDelta(48, 1, arma::fill::randn);
  MatType delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(8, 6);
  MatType targetDelta(4, 3);
  for (size_t i = 0; i < targetDelta.n_rows; i++)
  {
    for (size_t j = 0; j < targetDelta.n_cols; j++)
    {
      targetDelta.at(i, j) = (prevDelta.at(i * 2, j * 2)
          + prevDelta.at(i * 2 + 1, j * 2)
          + prevDelta.at(i * 2, j * 2 + 1)
          + prevDelta.at(i * 2 + 1, j * 2 + 1)) / 4;
    }
  }
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Repeat layer using block repeats.
 */
TEMPLATE_TEST_CASE("RepeatTestCaseB1", "[ANNLayerTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  // Input will be 4 x 3.
  MatType input(4, 3, arma::fill::randn);
  MatType target = arma::repmat(input, 2, 1);
  input.reshape(12, 1);
  target.reshape(24, 1);

  MatType output;
  // Output-Size should be 8 x 3.
  output.set_size(24, 1);

  RepeatType<MatType> module1({2, 1}, false);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 8);
  REQUIRE(module1.OutputDimensions()[1] == 3);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 24);
  REQUIRE(output.n_cols == 1);

  MatType prevDelta(24, 1, arma::fill::randn);
  MatType delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(8, 3);
  MatType targetDelta = (prevDelta.submat(0, 0, 3, 2) +
      prevDelta.submat(4, 0, 7, 2)) / 2;
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Replicate layer using block repeats.
 */
TEMPLATE_TEST_CASE("RepeatTestCaseB2", "[ANNLayerTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  // Input will be 4 x 3.
  MatType input(4, 3, arma::fill::randn);
  MatType target = arma::repmat(input, 1, 2);
  input.reshape(12, 1);
  target.reshape(24, 1);

  MatType output;
  // Output-Size should be 4 x 6.
  output.set_size(24, 1);

  RepeatType<MatType> module1({1, 2}, false);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 4);
  REQUIRE(module1.OutputDimensions()[1] == 6);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 24);
  REQUIRE(output.n_cols == 1);

  MatType prevDelta(24, 1, arma::fill::randn);
  MatType delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(4, 6);
  MatType targetDelta = (prevDelta.submat(0, 0, 3, 2)
                         + prevDelta.submat(0, 3, 3, 5)) / 2;
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

/**
 * Simple test for Replicate layer using block repeats.
 */
TEMPLATE_TEST_CASE("RepeatTestCaseB3", "[ANNLayerTest]", arma::mat, arma::fmat)
{
  using MatType = TestType;

  // Input will be 4 x 3.
  MatType input(4, 3, arma::fill::randn);
  MatType target = arma::repmat(input, 2, 2);
  input.reshape(12, 1);
  target.reshape(48, 1);

  MatType output;
  // Output-Size should be 8 x 6.
  output.set_size(48, 1);

  RepeatType<MatType> module1({2, 2}, false);
  module1.InputDimensions() = std::vector<size_t>({ 4, 3 });
  module1.ComputeOutputDimensions();
  REQUIRE(module1.OutputDimensions().size() == 2);
  REQUIRE(module1.OutputDimensions()[0] == 8);
  REQUIRE(module1.OutputDimensions()[1] == 6);
  module1.Forward(input, output);
  CheckMatrices(output, target, 1e-1);
  REQUIRE(output.n_elem == 48);
  REQUIRE(output.n_cols == 1);

  MatType prevDelta(48, 1, arma::fill::randn);
  MatType delta;
  delta.set_size(12, 1);
  module1.Backward(input, output, prevDelta, delta);
  prevDelta.reshape(8, 6);
  MatType targetDelta = (prevDelta.submat(0, 0, 3, 2)
                         + prevDelta.submat(4, 0, 7, 2)
                         + prevDelta.submat(0, 3, 3, 5)
                         + prevDelta.submat(4, 3, 7, 5)) / 4;
  targetDelta.reshape(12, 1);
  CheckMatrices(delta, targetDelta, 1e-1);
}

template <typename F> struct GradientBound {};
template <> struct GradientBound<arma::mat>
{
  static constexpr double eps = 1e-5;
  static constexpr double bound = 1e-4;
};
template <> struct GradientBound<arma::fmat>
{
  static constexpr double eps = 1e-3;
  static constexpr double bound = 3e-3;
};

/**
 * Numerical gradient test for MultiheadAttention layer.
 */
TEMPLATE_TEST_CASE("GradientRepeatTest", "[ANNLayerTest]", arma::mat,
    arma::fmat)
{
  using MatType = TestType;
  struct GradientFunction
  {
    GradientFunction(std::vector<size_t> multiples, bool interleave) :
        inputDimensions({3, 4}),
        multiples(std::move(multiples)),
        interleave(interleave),
        outputSize(8),
        vocabSize(5),
        batchSize(20)
    {
      size_t inputSize = std::accumulate(inputDimensions.begin(),
                                         inputDimensions.end(), 1,
                                         std::multiplies<>());
      input = arma::randu<MatType>(inputSize, batchSize);
      target = arma::zeros<MatType>(vocabSize, batchSize);
      for (size_t i = 0; i < target.n_elem; ++i)
      {
        const size_t label = RandInt(1, vocabSize);
        target(i) = label;
      }

      model = new FFN<NegativeLogLikelihoodType<MatType>,
                      RandomInitialization, MatType>();
      model->InputDimensions() = inputDimensions;
      model->ResetData(input, target);
      model->template Add<RepeatType<MatType>>(multiples, interleave);
      model->template Add<LinearType<MatType>>(vocabSize);
      model->template Add<LogSoftMaxType<MatType>>();
    }

    ~GradientFunction()
    {
      delete model;
    }

    double Gradient(MatType& gradient)
    {
      double error = model->Evaluate(model->Parameters(), 0, batchSize);
      model->Gradient(model->Parameters(), 0, gradient, batchSize);
      return error;
    }

    MatType& Parameters() { return model->Parameters(); }

    FFN<NegativeLogLikelihoodType<MatType>, RandomInitialization, MatType>*
        model;

    MatType input, target;
    const std::vector<size_t> inputDimensions;
    const std::vector<size_t> multiples;
    const bool interleave;
    const size_t outputSize;
    const size_t vocabSize;
    const size_t batchSize;
    size_t count;
  };

  GradientFunction fn1({2, 1}, true);
  GradientFunction fn2({2, 1}, false);
  GradientFunction fn3({1, 2}, true);
  GradientFunction fn4({1, 2}, false);
  GradientFunction fn5({2, 2}, true);
  GradientFunction fn6({2, 2}, false);

  double eps = GradientBound<MatType>::eps;
  double bound = GradientBound<MatType>::bound;

  CHECK(CheckGradient<GradientFunction, MatType>(fn1, eps) <= bound);
  CHECK(CheckGradient<GradientFunction, MatType>(fn2, eps) <= bound);
  CHECK(CheckGradient<GradientFunction, MatType>(fn3, eps) <= bound);
  CHECK(CheckGradient<GradientFunction, MatType>(fn4, eps) <= bound);
  CHECK(CheckGradient<GradientFunction, MatType>(fn5, eps) <= bound);
  CHECK(CheckGradient<GradientFunction, MatType>(fn6, eps) <= bound);
}
