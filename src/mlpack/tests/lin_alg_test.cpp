/**
 * @file lin_alg_test.cpp
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

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace arma;
using namespace mlpack;
using namespace mlpack::math;

BOOST_AUTO_TEST_SUITE(LinAlgTest);

/**
 * Test for linalg__private::Center().  There are no edge cases here, so we'll
 * just try it once for now.
 */
BOOST_AUTO_TEST_CASE(TestCenterA)
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
    for (int col = 0; col < 5; col++)
      BOOST_REQUIRE_CLOSE(tmp_out(row, col), (double) (col - 2) * row, 1e-5);
}

BOOST_AUTO_TEST_CASE(TestCenterB)
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
    for (int col = 0; col < 6; col++)
      BOOST_REQUIRE_CLOSE(tmp_out(row, col), (double) (col - 2.5) * row, 1e-5);
}

BOOST_AUTO_TEST_CASE(TestWhitenUsingEig)
{
  // After whitening using eigendecomposition, the covariance of
  // our matrix will be I (or something very close to that).
  // We are loading a matrix from an external file... bad choice.
  mat tmp, tmp_centered, whitened, whitening_matrix;

  data::Load("trainSet.csv", tmp);
  Center(tmp, tmp_centered);
  WhitenUsingEig(tmp_centered, whitened, whitening_matrix);

  mat newcov = ccov(whitened);
  for (int row = 0; row < 5; row++)
  {
    for (int col = 0; col < 5; col++)
    {
      if (row == col)
      {
        // diagonal will be 0 in the case of any zero-valued eigenvalues
        // (rank-deficient covariance case)
        if (std::abs(newcov(row, col)) > 1e-10)
          BOOST_REQUIRE_CLOSE(newcov(row, col), 1.0, 1e-10);
      }
      else
      {
        BOOST_REQUIRE_SMALL(newcov(row, col), 1e-10);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestOrthogonalize)
{
  // Generate a random matrix; then, orthogonalize it and test if it's
  // orthogonal.
  mat tmp, orth;
  data::Load("fake.csv", tmp);
  Orthogonalize(tmp, orth);

  // test orthogonality
  mat test = ccov(orth);
  double ival = test(0, 0);
  for (size_t row = 0; row < test.n_rows; row++)
  {
    for (size_t col = 0; col < test.n_cols; col++)
    {
      if (row == col)
      {
        if (std::abs(test(row, col)) > 1e-10)
          BOOST_REQUIRE_CLOSE(test(row, col), ival, 1e-10);
      }
      else
      {
        BOOST_REQUIRE_SMALL(test(row, col), 1e-10);
      }
    }
  }
}

// Test RemoveRows().
BOOST_AUTO_TEST_CASE(TestRemoveRows)
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
        BOOST_REQUIRE_EQUAL(accu(input.row(row) == output.row(outputRow)), 200);

        // Increment output row counter.
        ++outputRow;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(TestSvecSmat)
{
  arma::mat X(3, 3);
  X(0, 0) = 0; X(0, 1) = 1, X(0, 2) = 2;
  X(1, 0) = 1; X(1, 1) = 3, X(1, 2) = 4;
  X(2, 0) = 2; X(2, 1) = 4, X(2, 2) = 5;

  arma::vec sx;
  Svec(X, sx);
  BOOST_REQUIRE_CLOSE(sx(0), 0, 1e-7);
  BOOST_REQUIRE_CLOSE(sx(1), M_SQRT2 * 1., 1e-7);
  BOOST_REQUIRE_CLOSE(sx(2), M_SQRT2 * 2., 1e-7);
  BOOST_REQUIRE_CLOSE(sx(3), 3., 1e-7);
  BOOST_REQUIRE_CLOSE(sx(4), M_SQRT2 * 4., 1e-7);
  BOOST_REQUIRE_CLOSE(sx(5), 5., 1e-7);

  arma::mat Xtest;
  Smat(sx, Xtest);
  BOOST_REQUIRE_EQUAL(Xtest.n_rows, 3);
  BOOST_REQUIRE_EQUAL(Xtest.n_cols, 3);
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      BOOST_REQUIRE_CLOSE(X(i, j), Xtest(i, j), 1e-7);
}

BOOST_AUTO_TEST_CASE(TestSparseSvec)
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

  BOOST_REQUIRE_CLOSE(v0, 0, 1e-7);
  BOOST_REQUIRE_CLOSE(v1, M_SQRT2 * 1., 1e-7);
  BOOST_REQUIRE_CLOSE(v2, 0, 1e-7);
  BOOST_REQUIRE_CLOSE(v3, 0, 1e-7);
  BOOST_REQUIRE_CLOSE(v4, 0, 1e-7);
  BOOST_REQUIRE_CLOSE(v5, 0, 1e-7);
}

BOOST_AUTO_TEST_CASE(TestSymKronIdSimple)
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

  BOOST_REQUIRE_EQUAL(lhs.n_elem, rhs.n_elem);
  for (size_t j = 0; j < lhs.n_elem; j++)
    BOOST_REQUIRE_CLOSE(lhs(j), rhs(j), 1e-5);
}

BOOST_AUTO_TEST_CASE(TestSymKronId)
{
  const size_t n = 10;
  arma::mat A = arma::randu<arma::mat>(n, n);
  A += A.t();

  arma::mat Op;
  SymKronId(A, Op);

  for (size_t i = 0; i < 5; i++)
  {
    arma::mat X = arma::randu<arma::mat>(n, n);
    X += X.t();
    arma::vec sx;
    Svec(X, sx);

    const arma::vec lhs = Op * sx;
    const arma::mat Rhs = 0.5 * (A * X + X * A);
    arma::vec rhs;
    Svec(Rhs, rhs);

    BOOST_REQUIRE_EQUAL(lhs.n_elem, rhs.n_elem);
    for (size_t j = 0; j < lhs.n_elem; j++)
      BOOST_REQUIRE_CLOSE(lhs(j), rhs(j), 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
