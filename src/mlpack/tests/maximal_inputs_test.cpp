/**
 * @file tests/maximal_inputs_test.cpp
 * @author Ngap Wei Tham
 *
 * Test the MaximalInputs and ColumnsToBlocks functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_autoencoder/maximal_inputs.hpp>

#include "catch.hpp"

using namespace mlpack;

arma::mat CreateMaximalInput()
{
  arma::mat w1(2, 4);
  w1 = { {0, 1, 2, 3},
         {4, 5, 6, 7} };

  arma::mat input(5, 5);
  input.submat(0, 0, 1, 3) = w1;

  arma::mat maximalInputs;
  MaximalInputs(input, maximalInputs);

  return maximalInputs;
}

void TestResults(const arma::mat&actualResult, const arma::mat& expectResult)
{
  REQUIRE(expectResult.n_rows == actualResult.n_rows);
  REQUIRE(expectResult.n_cols == actualResult.n_cols);

  for (size_t i = 0; i != expectResult.n_elem; ++i)
  {
    REQUIRE(expectResult[i] == Approx(actualResult[i]).epsilon(1e-4));
  }
}

TEST_CASE("ColumnToBlocksEvaluate", "[MaximalInputsTest]")
{
  arma::mat output;
  ColumnsToBlocks ctb(1, 2);
  ctb.Transform(CreateMaximalInput(), output);

  arma::mat matlabResults;
  matlabResults = { { -1,       -1,       -1, -1,      -1,      -1, -1 },
                    { -1,       -1, -0.42857, -1, 0.14286, 0.71429, -1 },
                    { -1, -0.71429, -0.14286, -1, 0.42857,       1, -1 },
                    { -1,       -1,       -1, -1,      -1,      -1, -1 } };

  TestResults(output, matlabResults);
}

TEST_CASE("ColumnToBlocksChangeBlockSize", "[MaximalInputsTest]")
{
  arma::mat output;
  ColumnsToBlocks ctb(1, 2);
  ctb.BlockWidth(4);
  ctb.BlockHeight(1);
  ctb.BufValue(-3);
  ctb.Transform(CreateMaximalInput(), output);

  arma::mat matlabResults;
  matlabResults = { { -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3 },
                    { -3, -1, -0.71429, -0.42857, -0.14286, -3, 0.14286,
                     0.42857, 0.71429,  1, -3 },
                    { -3, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3 } };

  TestResults(output, matlabResults);
}
