/**
 * @file lin_alg_test.cpp
 * @author Ryan Curtin
 *
 * Simple tests for things in the linalg__private namespace.
 * Partly so I can be sure that my changes are working.
 * Move to boost unit testing framework at some point.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/math/lin_alg.hpp>

#include <boost/test/unit_test.hpp>

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

BOOST_AUTO_TEST_SUITE_END();
