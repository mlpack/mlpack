/**
 * @file tests/lin_alg_test.cpp
 * @author Ryan Curtin
 *
 * Simple tests for things in the linalg__private namespace.
 * Partly so I can be sure that my changes are working.
 * Move to boost unit testing framework at some point.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace arma;
using namespace mlpack;

/**
 * Test for linalg__private::Center().  There are no edge cases here, so we'll
 * just try it once for now.
 */
TEST_CASE("TestCenterA", "[LinAlgTest]")
{
  mat tmp(5, 5);
  // [[0  0  0  0  0]
  //  [1  2  3  4  5]
  //  [2  4  6  8  10]
  //  [3  6  9  12 15]
  //  [4  8  12 16 20]]
  for (int row = 0; row < 5; row++)
    for (int col = 0; col < 5; col++)
      tmp(row, col) = row * (col + 1);

  mat tmp_out;
  Center(tmp, tmp_out);

  // average should be
  // [[0 3 6 9 12]]'
  // so result should be
  // [[ 0  0  0  0  0]
  //  [-2 -1  0  1  2 ]
  //  [-4 -2  0  2  4 ]
  //  [-6 -3  0  3  6 ]
  //  [-8 -4  0  4  8]]
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      REQUIRE(tmp_out(row, col) ==
          Approx((double) (col - 2) * row).epsilon(1e-7));
    }
  }
}

TEST_CASE("TestCenterB", "[LinAlgTest]")
{
  mat tmp(5, 6);
  for (int row = 0; row < 5; row++)
    for (int col = 0; col < 6; col++)
      tmp(row, col) = row * (col + 1);

  mat tmp_out;
  Center(tmp, tmp_out);

  // average should be
  // [[0 3.5 7 10.5 14]]'
  // so result should be
  // [[ 0    0    0   0   0   0  ]
  //  [-2.5 -1.5 -0.5 0.5 1.5 2.5]
  //  [-5   -3   -1   1   3   5  ]
  //  [-7.5 -4.5 -1.5 1.5 1.5 4.5]
  //  [-10  -6   -2   2   6   10 ]]
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 6; col++)
    {
      REQUIRE(tmp_out(row, col) ==
          Approx((double) (col - 2.5) * row).epsilon(1e-7));
    }
  }
}

// Test RemoveRows().
TEST_CASE("TestRemoveRows", "[LinAlgTest]")
{
  // Run this test several times.
  for (size_t run = 0; run < 10; ++run)
  {
    arma::mat input;
    input.randu(200, 200);

    // Now pick some random numbers.
    std::vector<size_t> rowsToRemove;
    size_t row = 0;
    while (row < 200)
    {
      row += RandInt(1, (2 * (run + 1) + 1));
      if (row < 200)
      {
        rowsToRemove.push_back(row);
      }
    }

    // Ensure we're not about to remove every single row.
    if (rowsToRemove.size() == 10)
    {
      rowsToRemove.erase(rowsToRemove.begin() + 4); // Random choice to remove.
    }

    arma::mat output;
    RemoveRows(input, rowsToRemove, output);

    // Now check that the output is right.
    size_t outputRow = 0;
    size_t skipIndex = 0;

    for (row = 0; row < 200; ++row)
    {
      // Was this row supposed to be removed?  If so skip it.
      if ((skipIndex < rowsToRemove.size()) && (rowsToRemove[skipIndex] == row))
      {
        ++skipIndex;
      }
      else
      {
        // Compare.
        REQUIRE(accu(input.row(row) == output.row(outputRow)) == 200);

        // Increment output row counter.
        ++outputRow;
      }
    }
  }
}
