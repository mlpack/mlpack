/**
 * @file tests/convolution_test.cpp
 * @author Shangtong Zhang
 * @author Marcus Edel
 *
 * Tests for various convolution strategies.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/convolution_rules/convolution_rules.hpp>

#include "../serialization.hpp"
#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

/*
 * Implementation of the convolution function test with custom stride and
 * dilation.  Some convolution types only work with 1 stride and dilation.
 *
 * @param input Input used to perform the convolution.
 * @param filter Filter used to perform the convolution.
 * @param output The reference output data that contains the results of the
 * convolution.
 * @param strideH Height stride parameter.
 * @param strideW Width stride parameter.
 * @param dilationH Height dilation parameter.
 * @param dilationW Width dilation parameter.
 *
 * @tparam ConvolutionFunction Convolution function used for the check.
 */
template<class ConvolutionFunction>
void Convolution2DMethodTest(const arma::mat input,
                             const arma::mat filter,
                             const arma::mat output,
                             const size_t strideW = 1,
                             const size_t strideH = 1,
                             const size_t dilationW = 1,
                             const size_t dilationH = 1)
{
  arma::mat convOutput;
  ConvolutionFunction::Convolution(input, filter, convOutput, strideW, strideH,
      dilationW, dilationH);

  // Check the output dimension.
  bool b = (convOutput.n_rows == output.n_rows) &&
      (convOutput.n_cols == output.n_cols);
  REQUIRE(b == 1);

  const double* outputPtr = output.memptr();
  const double* convOutputPtr = convOutput.memptr();

  for (size_t i = 0; i < output.n_elem; ++i, outputPtr++, convOutputPtr++)
    REQUIRE(*outputPtr == Approx(*convOutputPtr).epsilon(1e-5));
}

/*
 * Implementation of the convolution function test using 3rd order tensors.
 *
 * @param input Input used to perform the convolution.
 * @param filter Filter used to perform the convolution.
 * @param output The reference output data that contains the results of the
 * convolution.
 *
 * @tparam ConvolutionFunction Convolution function used for the check.
 */
template<class ConvolutionFunction>
void Convolution3DMethodTest(const arma::cube input,
                             const arma::cube filter,
                             const arma::cube output)
{
  arma::cube convOutput;
  ConvolutionFunction::Convolution(input, filter, convOutput);

  // Check the output dimension.
  bool b = (convOutput.n_rows == output.n_rows) &&
      (convOutput.n_cols == output.n_cols &&
      convOutput.n_slices == output.n_slices);
  REQUIRE(b == 1);

  const double* outputPtr = output.memptr();
  const double* convOutputPtr = convOutput.memptr();

  for (size_t i = 0; i < output.n_elem; ++i, outputPtr++, convOutputPtr++)
    REQUIRE(*outputPtr == Approx(*convOutputPtr).epsilon(1e-5));
}

/*
 * Implementation of the convolution function test using dense matrix as input
 * and a 3rd order tensors as filter and output (batch modus).
 *
 * @param input Input used to perform the convolution.
 * @param filter Filter used to perform the convolution.
 * @param output The reference output data that contains the results of the
 * convolution.
 *
 * @tparam ConvolutionFunction Convolution function used for the check.
 */
template<class ConvolutionFunction>
void ConvolutionMethodBatchTest(const arma::mat input,
                                const arma::cube filter,
                                const arma::cube output)
{
  arma::cube convOutput;
  ConvolutionFunction::Convolution(input, filter, convOutput);

  // Check the output dimension.
  bool b = (convOutput.n_rows == output.n_rows) &&
      (convOutput.n_cols == output.n_cols &&
      convOutput.n_slices == output.n_slices);
  REQUIRE(b == 1);

  const double* outputPtr = output.memptr();
  const double* convOutputPtr = convOutput.memptr();

  for (size_t i = 0; i < output.n_elem; ++i, outputPtr++, convOutputPtr++)
    REQUIRE(*outputPtr == Approx(*convOutputPtr).epsilon(1e-5));
}

/**
 * Test the convolution (valid) methods.
 */
TEST_CASE("ValidConvolution2DTest", "[ConvolutionTest]")
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { {  1, 0, -1 },
             {  0, 1,  0 },
             { -1, 0,  1 } };

  output = { { -3, -2 },
             {  8, -3 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<ValidConvolution> >(input, filter,
      output);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<ValidConvolution> >(input, filter,
      output);
}

/**
 * Test the convolution (full) methods.
 */
TEST_CASE("FullConvolution2DTest", "[ConvolutionTest]")
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { {  1, 0, -1 },
             {  1, 1,  1 },
             { -1, 0,  1 } };

  output = { {  1,  2,  2,  2, -3, -4 },
             {  5,  4,  4, 11,  5,  1 },
             {  6,  7,  3,  2,  7,  5 },
             {  1,  9, 12,  3,  1,  4 },
             { -1,  1, 11, 10,  6,  3 },
             { -2, -3, -2,  2,  4,  1 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output);
}

/**
 * Test the convolution (valid) methods using 3rd order tensors.
 */
TEST_CASE("ValidConvolution3DTest", "[ConvolutionTest]")
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { {  1, 0, -1 },
             {  0, 1,  0 },
             { -1, 0,  1 } };

  output = { { -6, -4 },
             { 16, -6 } };

  arma::cube inputCube(input.n_rows, input.n_cols, 2);
  inputCube.slice(0) = input;
  inputCube.slice(1) = input;

  arma::cube filterCube(filter.n_rows, filter.n_cols, 4);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;
  filterCube.slice(2) = filter;
  filterCube.slice(3) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  Convolution3DMethodTest<NaiveConvolution<ValidConvolution> >(inputCube,
      filterCube, outputCube);

  // Perform the convolution using im2col.
  Convolution3DMethodTest<Im2ColConvolution<ValidConvolution> >(inputCube,
      filterCube, outputCube);
}

/**
 * Test the convolution (full) methods using 3rd order tensors.
 */
TEST_CASE("FullConvolution3DTest", "[ConvolutionTest]")
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { {  1, 0, -1 },
             {  1, 1,  1 },
             { -1, 0,  1 } };

  output = { {  2,  4,  4,  4, -6, -8 },
             { 10,  8,  8, 22, 10,  2 },
             { 12, 14,  6,  4, 14, 10 },
             {  2, 18, 24,  6,  2,  8 },
             { -2,  2, 22, 20, 12,  6 },
             { -4, -6, -4,  4,  8,  2 } };

  arma::cube inputCube(input.n_rows, input.n_cols, 2);
  inputCube.slice(0) = input;
  inputCube.slice(1) = input;

  arma::cube filterCube(filter.n_rows, filter.n_cols, 4);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;
  filterCube.slice(2) = filter;
  filterCube.slice(3) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  Convolution3DMethodTest<NaiveConvolution<FullConvolution> >(inputCube,
      filterCube, outputCube);

  // Perform the convolution using im2col.
  Convolution3DMethodTest<Im2ColConvolution<FullConvolution> >(inputCube,
      filterCube, outputCube);
}

/**
 * Test the convolution (valid) methods using dense matrix as input and a 3rd
 * order tensors as filter and output (batch modus).
 */
TEST_CASE("ValidConvolutionBatchTest", "[ConvolutionTest]")
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { {  1, 0, -1 },
             {  0, 1,  0 },
             { -1, 0,  1 } };

  output = { { -3, -2 },
             {  8, -3 } };

  arma::cube filterCube(filter.n_rows, filter.n_cols, 2);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  ConvolutionMethodBatchTest<NaiveConvolution<ValidConvolution> >(input,
      filterCube, outputCube);

  // Perform the convolution using im2col.
  ConvolutionMethodBatchTest<Im2ColConvolution<ValidConvolution> >(input,
      filterCube, outputCube);
}

/**
 * Test the convolution (full) methods using dense matrix as input and a 3rd
 * order tensors as filter and output (batch modus).
 */
TEST_CASE("FullConvolutionBatchTest", "[ConvolutionTest]")
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { {  1, 0, -1 },
             {  1, 1,  1 },
             { -1, 0,  1 } };

  output = { {  1,  2,  2,  2, -3, -4 },
             {  5,  4,  4, 11,  5,  1 },
             {  6,  7,  3,  2,  7,  5 },
             {  1,  9, 12,  3,  1,  4 },
             { -1,  1, 11, 10,  6,  3 },
             { -2, -3, -2,  2,  4,  1 } };

  arma::cube filterCube(filter.n_rows, filter.n_cols, 2);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  ConvolutionMethodBatchTest<NaiveConvolution<FullConvolution> >(input,
      filterCube, outputCube);

  // Perform the convolution using im2col.
  ConvolutionMethodBatchTest<Im2ColConvolution<FullConvolution> >(input,
      filterCube, outputCube);
}

/**
 * Test that non-stride-1 convolution works the same as stride-1 convolution on
 * a smaller matrix.
 */
TEST_CASE("Stride2ConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 1, -4 },
             { -1, -4, 1 },
             { -2, -1, 1 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 2, 2, 1, 1);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 2, 2, 1, 1);
}

TEST_CASE("Stride3ConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 1 },
             { -1, -4 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 3, 3, 1, 1);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 3, 3, 1, 1);
}

TEST_CASE("UnequalStrideConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 1 },
             { -1, 0 },
             { -2, 3 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 3, 2, 1, 1);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 3, 2, 1, 1);
}

TEST_CASE("Dilation2ConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 2, 2, 2, -3, -4 },
             { 4, 1, -2, 2, -2, -3 },
             { 2, 2, -4, -4, 2, 2 },
             { -2, 2, 4, -4, -2, 2 },
             { -3, -4, 2, 2, 1, 2 },
             { -2, -3, -2, 2, 4, 1 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 1, 1, 2, 2);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 1, 1, 2, 2);
}

TEST_CASE("Dilation3ConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 2, 3, 3, -2, -3, -4 },
             { 4, 1, 2, -1, -1, -2, -3 },
             { 3, 4, 1, -1, -4, -1, -2 },
             { 1, 1, 1, -4, -1, -1, 3 },
             { -4, -1, -2, 1, 1, 2, 3 },
             { -3, -4, -1, 1, 4, 1, 2 },
             { -2, -3, -4, 1, 3, 4, 1 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 1, 1, 3, 3);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 1, 1, 3, 3);
}

TEST_CASE("UnequalDilationConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 2, 3, 3, -2, -3, -4 },
             { 4, 1, 2, -1, -1, -2, -3 },
             { 2, 2, -2, -4, -2, 2, 2 },
             { -2, 2, 2, 0, -2, -2, 2 },
             { -3, -4, -1, 1, 4, 1, 2 },
             { -2, -3, -4, 1, 3, 4, 1 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 1, 1, 3, 2);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 1, 1, 3, 2);
}

TEST_CASE("DilationAndStrideConvolutionTest", "[ConvolutionTest]")
{
  // Generate dataset.
  arma::mat input, filter, output;
  input = { { 1, 2, 3, 4 },
            { 4, 1, 2, 3 },
            { 3, 4, 1, 2 },
            { 2, 3, 4, 1 } };

  filter = { { 1, -1 },
             { -1, 1 } };

  output = { { 1, 2, -3 },
             { 2, -4, 2 },
             { -3, 2, 1 } };

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output, 2, 2, 2, 2);

  // Perform the convolution using im2col.
  Convolution2DMethodTest<Im2ColConvolution<FullConvolution> >(input, filter,
      output, 2, 2, 2, 2);
}
