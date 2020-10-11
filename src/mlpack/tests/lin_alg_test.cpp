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
#include <mlpack/core/math/lin_alg.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::math;

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

TEST_CASE("TestOrthogonalize", "[LinAlgTest]")
{
  // Generate a random matrix; then, orthogonalize it and test if it's
  // orthogonal.
  mat tmp, orth;
  data::Load("fake.csv", tmp);
  Orthogonalize(tmp, orth);

  // test orthogonality
  mat test = mlpack::math::ColumnCovariance(orth);
  double ival = test(0, 0);
  for (size_t row = 0; row < test.n_rows; row++)
  {
    for (size_t col = 0; col < test.n_cols; col++)
    {
      if (row == col)
      {
        if (std::abs(test(row, col)) > 1e-10)
          REQUIRE(test(row, col) == Approx(ival).epsilon(1e-11));
      }
      else
      {
        REQUIRE(test(row, col) == Approx(0.0).margin(1e-10));
      }
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

TEST_CASE("TestSvecSmat", "[LinAlgTest]")
{
  arma::mat X(3, 3);
  X(0, 0) = 0; X(0, 1) = 1, X(0, 2) = 2;
  X(1, 0) = 1; X(1, 1) = 3, X(1, 2) = 4;
  X(2, 0) = 2; X(2, 1) = 4, X(2, 2) = 5;

  arma::vec sx;
  Svec(X, sx);
  REQUIRE(sx(0) == Approx(0).epsilon(1e-9));
  REQUIRE(sx(1) == Approx(M_SQRT2 * 1.).epsilon(1e-9));
  REQUIRE(sx(2) == Approx(M_SQRT2 * 2.).epsilon(1e-9));
  REQUIRE(sx(3) == Approx(3.).epsilon(1e-9));
  REQUIRE(sx(4) == Approx(M_SQRT2 * 4.).epsilon(1e-9));
  REQUIRE(sx(5) == Approx(5.).epsilon(1e-9));

  arma::mat Xtest;
  Smat(sx, Xtest);
  REQUIRE(Xtest.n_rows == 3);
  REQUIRE(Xtest.n_cols == 3);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      REQUIRE(X(i, j) == Approx(Xtest(i, j)).epsilon(1e-9));

}

TEST_CASE("TestSparseSvec", "[LinAlgTest]")
{
  arma::sp_mat X;
  X.zeros(3, 3);
  X(1, 0) = X(0, 1) = 1;

  arma::sp_vec sx;
  Svec(X, sx);

  const double v0 = sx(0);
  const double v1 = sx(1);
  const double v2 = sx(2);
  const double v3 = sx(3);
  const double v4 = sx(4);
  const double v5 = sx(5);

  REQUIRE(v0 == Approx(0).epsilon(1e-9));
  REQUIRE(v1 == Approx(M_SQRT2 * 1.).epsilon(1e-9));
  REQUIRE(v2 == Approx(0).epsilon(1e-9));
  REQUIRE(v3 == Approx(0).epsilon(1e-9));
  REQUIRE(v4 == Approx(0).epsilon(1e-9));
  REQUIRE(v5 == Approx(0).epsilon(1e-9));
}

TEST_CASE("TestSymKronIdSimple", "[LinAlgTest]")
{
  arma::mat A(3, 3);
  A(0, 0) = 1; A(0, 1) = 2, A(0, 2) = 3;
  A(1, 0) = 2; A(1, 1) = 4, A(1, 2) = 5;
  A(2, 0) = 3; A(2, 1) = 5, A(2, 2) = 6;
  arma::mat Op;
  SymKronId(A, Op);

  const arma::mat X = A + arma::ones<arma::mat>(3, 3);
  arma::vec sx;
  Svec(X, sx);

  const arma::vec lhs = Op * sx;
  const arma::mat Rhs = 0.5 * (A * X + X * A);
  arma::vec rhs;
  Svec(Rhs, rhs);

  REQUIRE(lhs.n_elem == rhs.n_elem);
  for (size_t j = 0; j < lhs.n_elem; ++j)
    REQUIRE(lhs(j) == Approx(rhs(j)).epsilon(1e-7));
}

TEST_CASE("TestSymKronId", "[LinAlgTest]")
{
  const size_t n = 10;
  arma::mat A = arma::randu<arma::mat>(n, n);
  A += A.t();

  arma::mat Op;
  SymKronId(A, Op);

  for (size_t i = 0; i < 5; ++i)
  {
    arma::mat X = arma::randu<arma::mat>(n, n);
    X += X.t();
    arma::vec sx;
    Svec(X, sx);

    const arma::vec lhs = Op * sx;
    const arma::mat Rhs = 0.5 * (A * X + X * A);
    arma::vec rhs;
    Svec(Rhs, rhs);

    REQUIRE(lhs.n_elem == rhs.n_elem);
    for (size_t j = 0; j < lhs.n_elem; ++j)
      REQUIRE(lhs(j) == Approx(rhs(j)).epsilon(1e-7));
  }
}
