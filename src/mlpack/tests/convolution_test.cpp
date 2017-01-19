/**
 * @file convolution_test.cpp
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

#include <mlpack/methods/ann/convolution_rules/border_modes.hpp>
#include <mlpack/methods/ann/convolution_rules/naive_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/fft_convolution.hpp>
#include <mlpack/methods/ann/convolution_rules/svd_convolution.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ConvolutionTest);

/*
 * Implementation of the convolution function test.
 *
 * @param input Input used to perform the convolution.
 * @param filter Filter used to perform the conolution.
 * @param output The reference output data that contains the results of the
 * convolution.
 *
 * @tparam ConvolutionFunction Convolution function used for the check.
 */
template<class ConvolutionFunction>
void Convolution2DMethodTest(const arma::mat input,
                             const arma::mat filter,
                             const arma::mat output)
{
  arma::mat convOutput;
  ConvolutionFunction::Convolution(input, filter, convOutput);

  // Check the outut dimension.
  bool b = (convOutput.n_rows == output.n_rows) &&
      (convOutput.n_cols == output.n_cols);
  BOOST_REQUIRE_EQUAL(b, 1);

  const double* outputPtr = output.memptr();
  const double* convOutputPtr = convOutput.memptr();

  for (size_t i = 0; i < output.n_elem; i++, outputPtr++, convOutputPtr++)
    BOOST_REQUIRE_CLOSE(*outputPtr, *convOutputPtr, 1e-3);
}

/*
 * Implementation of the convolution function test using 3rd order tensors.
 *
 * @param input Input used to perform the convolution.
 * @param filter Filter used to perform the conolution.
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

  // Check the outut dimension.
  bool b = (convOutput.n_rows == output.n_rows) &&
      (convOutput.n_cols == output.n_cols &&
      convOutput.n_slices == output.n_slices);
  BOOST_REQUIRE_EQUAL(b, 1);

  const double* outputPtr = output.memptr();
  const double* convOutputPtr = convOutput.memptr();

  for (size_t i = 0; i < output.n_elem; i++, outputPtr++, convOutputPtr++)
    BOOST_REQUIRE_CLOSE(*outputPtr, *convOutputPtr, 1e-3);
}

/*
 * Implementation of the convolution function test using dense matrix as input
 * and a 3rd order tensors as filter and output (batch modus).
 *
 * @param input Input used to perform the convolution.
 * @param filter Filter used to perform the conolution.
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

  // Check the outut dimension.
  bool b = (convOutput.n_rows == output.n_rows) &&
      (convOutput.n_cols == output.n_cols &&
      convOutput.n_slices == output.n_slices);
  BOOST_REQUIRE_EQUAL(b, 1);

  const double* outputPtr = output.memptr();
  const double* convOutputPtr = convOutput.memptr();

  for (size_t i = 0; i < output.n_elem; i++, outputPtr++, convOutputPtr++)
    BOOST_REQUIRE_CLOSE(*outputPtr, *convOutputPtr, 1e-3);
}

/**
 * Test the convolution (valid) methods.
 */
BOOST_AUTO_TEST_CASE(ValidConvolution2DTest)
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input << 1 << 2 << 3 << 4 << arma::endr
        << 4 << 1 << 2 << 3 << arma::endr
        << 3 << 4 << 1 << 2 << arma::endr
        << 2 << 3 << 4 << 1;

  filter << 1 << 0 << -1 << arma::endr
         << 0 << 1 << 0 << arma::endr
         << -1 << 0 << 1;

  output << -3 << -2 << arma::endr
         << 8 << -3;

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<ValidConvolution> >(input, filter,
      output);

  // Perform the convolution trough fft.
  Convolution2DMethodTest<FFTConvolution<ValidConvolution> >(input, filter,
      output);

  // Perform the convolution using singular value decomposition to
  // speeded up the computation.
  Convolution2DMethodTest<SVDConvolution<ValidConvolution> >(input, filter,
      output);
}

/**
 * Test the convolution (full) methods.
 */
BOOST_AUTO_TEST_CASE(FullConvolution2DTest)
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input << 1 << 2 << 3 << 4 << arma::endr
        << 4 << 1 << 2 << 3 << arma::endr
        << 3 << 4 << 1 << 2 << arma::endr
        << 2 << 3 << 4 << 1;

  filter << 1 << 0 << -1 << arma::endr
         << 1 << 1 << 1 << arma::endr
         << -1 << 0 << 1;

  output << 1 << 2 << 2 << 2 << -3 << -4 << arma::endr
         << 5 << 4 << 4 << 11 << 5 << 1 << arma::endr
         << 6 << 7 << 3 << 2 << 7 << 5 << arma::endr
         << 1 << 9 << 12 << 3 << 1 << 4 << arma::endr
         << -1 << 1 << 11 << 10 << 6 << 3 << arma::endr
         << -2 << -3 << -2 << 2 << 4 << 1;

  // Perform the naive convolution approach.
  Convolution2DMethodTest<NaiveConvolution<FullConvolution> >(input, filter,
      output);

  // Perform the convolution trough fft.
  Convolution2DMethodTest<FFTConvolution<FullConvolution> >(input, filter,
      output);

  // Perform the convolution using singular value decomposition to
  // speeded up the computation.
  Convolution2DMethodTest<SVDConvolution<FullConvolution> >(input, filter,
      output);
}

/**
 * Test the convolution (valid) methods using 3rd order tensors.
 */
BOOST_AUTO_TEST_CASE(ValidConvolution3DTest)
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input << 1 << 2 << 3 << 4 << arma::endr
        << 4 << 1 << 2 << 3 << arma::endr
        << 3 << 4 << 1 << 2 << arma::endr
        << 2 << 3 << 4 << 1;

  filter << 1 << 0 << -1 << arma::endr
         << 0 << 1 << 0 << arma::endr
         << -1 << 0 << 1;

  output << -3 << -2 << arma::endr
         << 8 << -3;

  arma::cube inputCube(input.n_rows, input.n_cols, 2);
  inputCube.slice(0) = input;
  inputCube.slice(1) = input;

  arma::cube filterCube(filter.n_rows, filter.n_cols, 2);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  Convolution3DMethodTest<NaiveConvolution<ValidConvolution> >(inputCube,
      filterCube, outputCube);

  // Perform the convolution trough fft.
  Convolution3DMethodTest<FFTConvolution<ValidConvolution> >(inputCube,
      filterCube, outputCube);

  // Perform the convolution using using the singular value decomposition to
  // speeded up the computation.
  Convolution3DMethodTest<SVDConvolution<ValidConvolution> >(inputCube,
      filterCube, outputCube);
}

/**
 * Test the convolution (full) methods using 3rd order tensors.
 */
BOOST_AUTO_TEST_CASE(FullConvolution3DTest)
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input << 1 << 2 << 3 << 4 << arma::endr
        << 4 << 1 << 2 << 3 << arma::endr
        << 3 << 4 << 1 << 2 << arma::endr
        << 2 << 3 << 4 << 1;

  filter << 1 << 0 << -1 << arma::endr
         << 1 << 1 << 1 << arma::endr
         << -1 << 0 << 1;

  output << 1 << 2 << 2 << 2 << -3 << -4 << arma::endr
         << 5 << 4 << 4 << 11 << 5 << 1 << arma::endr
         << 6 << 7 << 3 << 2 << 7 << 5 << arma::endr
         << 1 << 9 << 12 << 3 << 1 << 4 << arma::endr
         << -1 << 1 << 11 << 10 << 6 << 3 << arma::endr
         << -2 << -3 << -2 << 2 << 4 << 1;

  arma::cube inputCube(input.n_rows, input.n_cols, 2);
  inputCube.slice(0) = input;
  inputCube.slice(1) = input;

  arma::cube filterCube(filter.n_rows, filter.n_cols, 2);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  Convolution3DMethodTest<NaiveConvolution<FullConvolution> >(inputCube,
      filterCube, outputCube);

  // Perform the convolution trough fft.
  Convolution3DMethodTest<FFTConvolution<FullConvolution> >(inputCube,
      filterCube, outputCube);

  // Perform the convolution using using the singular value decomposition to
  // speeded up the computation.
  Convolution3DMethodTest<SVDConvolution<FullConvolution> >(inputCube,
      filterCube, outputCube);
}

/**
 * Test the convolution (valid) methods using dense matrix as input and a 3rd
 * order tensors as filter and output (batch modus).
 */
BOOST_AUTO_TEST_CASE(ValidConvolutionBatchTest)
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input << 1 << 2 << 3 << 4 << arma::endr
        << 4 << 1 << 2 << 3 << arma::endr
        << 3 << 4 << 1 << 2 << arma::endr
        << 2 << 3 << 4 << 1;

  filter << 1 << 0 << -1 << arma::endr
         << 0 << 1 << 0 << arma::endr
         << -1 << 0 << 1;

  output << -3 << -2 << arma::endr
         << 8 << -3;

  arma::cube filterCube(filter.n_rows, filter.n_cols, 2);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  ConvolutionMethodBatchTest<NaiveConvolution<ValidConvolution> >(input,
      filterCube, outputCube);

  // Perform the convolution trough fft.
  ConvolutionMethodBatchTest<FFTConvolution<ValidConvolution> >(input,
      filterCube, outputCube);

  // Perform the convolution using using the singular value decomposition to
  // speeded up the computation.
  ConvolutionMethodBatchTest<SVDConvolution<ValidConvolution> >(input,
      filterCube, outputCube);
}

/**
 * Test the convolution (full) methods using dense matrix as input and a 3rd
 * order tensors as filter and output (batch modus).
 */
BOOST_AUTO_TEST_CASE(FullConvolutionBatchTest)
{
  // Generate dataset for convolution function tests.
  arma::mat input, filter, output;
  input << 1 << 2 << 3 << 4 << arma::endr
        << 4 << 1 << 2 << 3 << arma::endr
        << 3 << 4 << 1 << 2 << arma::endr
        << 2 << 3 << 4 << 1;

  filter << 1 << 0 << -1 << arma::endr
         << 1 << 1 << 1 << arma::endr
         << -1 << 0 << 1;

  output << 1 << 2 << 2 << 2 << -3 << -4 << arma::endr
         << 5 << 4 << 4 << 11 << 5 << 1 << arma::endr
         << 6 << 7 << 3 << 2 << 7 << 5 << arma::endr
         << 1 << 9 << 12 << 3 << 1 << 4 << arma::endr
         << -1 << 1 << 11 << 10 << 6 << 3 << arma::endr
         << -2 << -3 << -2 << 2 << 4 << 1;

  arma::cube filterCube(filter.n_rows, filter.n_cols, 2);
  filterCube.slice(0) = filter;
  filterCube.slice(1) = filter;

  arma::cube outputCube(output.n_rows, output.n_cols, 2);
  outputCube.slice(0) = output;
  outputCube.slice(1) = output;

  // Perform the naive convolution approach.
  ConvolutionMethodBatchTest<NaiveConvolution<FullConvolution> >(input,
      filterCube, outputCube);

  // Perform the convolution trough fft.
  ConvolutionMethodBatchTest<FFTConvolution<FullConvolution> >(input,
      filterCube, outputCube);

  // Perform the convolution using using the singular value decomposition to
  // speeded up the computation.
  ConvolutionMethodBatchTest<SVDConvolution<FullConvolution> >(input,
      filterCube, outputCube);
}

BOOST_AUTO_TEST_SUITE_END();
