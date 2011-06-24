/***
 * lin_alg_test.cc
 *
 * Simple tests for things in the linalg__private namespace.
 * Partly so I can be sure that my changes are working.
 * Move to boost unit testing framework at some point.
 *
 * @author Ryan Curtin
 */
#include <fastlib/fastlib.h>
#include <armadillo>
#include <fastlib/base/arma_extend.h>

#include "lin_alg.h"

using namespace arma;
using namespace linalg__private;


#define BOOST_TEST_MODULE linAlgTest
#include <boost/test/unit_test.hpp>

/***
 * Test for linalg__private::Center().  There are no edge cases here, so we'll
 * just try it once for now.
 */
bool test_Center() {
  mat tmp(5, 5);
  // [[0  0  0  0  0]
  //  [1  2  3  4  5]
  //  [2  4  6  8  10]
  //  [3  6  9  12 15]
  //  [4  8  12 16 20]]
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 5; col++)
      tmp(row, col) = row * (col + 1);
  }

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
  for (int row = 0; row < 5; row++) {
    for (int col = 0; col < 5; col++) {
      BOOST_REQUIRE_CLOSE(tmp_out(row, col), (col - 2) * row, 1e-5);
    }
  }
  
  mat tmp2(5, 6);
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 6; col++)
      tmp2(row, col) = row * (col + 1);
  }
  Center(tmp2, tmp_out);

  // average should be
  // [[0 3.5 7 10.5 14]]'
  // so result should be
  // [[ 0    0    0   0   0   0  ]
  //  [-2.5 -1.5 -0.5 0.5 1.5 2.5]
  //  [-5   -3   -1   1   3   5  ]
  //  [-7.5 -4.5 -1.5 1.5 1.5 4.5]
  //  [-10  -6   -2   2   6   10 ]]
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 6; col++) {
      BOOST_REQUIRE_CLOSE(tmp_out(row, col), (col - 2.5) * row, 1e-5);
    }
  }
  
  return true;
}

bool test_WhitenUsingEig() {
  // After whitening using eigendecomposition, the covariance of
  // our matrix will be I (or something very close to that).
  // We are loading a matrix from an external file... bad choice.
  mat tmp, tmp_centered, whitened, whitening_matrix;

  data::Load("fake.arff", tmp);
  Center(tmp, tmp_centered);
  WhitenUsingEig(tmp_centered, whitened, whitening_matrix);
 
  mat newcov = ccov(whitened);
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 5; col++) {
      if(row == col) {
        // diagonal will be 0 in the case of any zero-valued eigenvalues
        // (rank-deficient covariance case)
        if(std::abs(newcov(row, col) - 1) > 1e-12 && newcov(row, col) != 0)
          return false;
      } else {
        if(std::abs(newcov(row, col)) > 1e-12)
          return false;
      }
    }
  }

  return true;
}

bool test_Orthogonalize() {
  // Generate a random matrix; then, orthogonalize it and test if it's
  // orthogonal.
  mat tmp, orth;
  data::Load("fake.arff", tmp);
  Orthogonalize(tmp, orth);

  // test orthogonality
  mat test = ccov(orth);
  double ival = test(0, 0);
  for(index_t row = 0; row < test.n_rows; row++) {
    for(index_t col = 0; col < test.n_cols; col++) {
      if(row == col) {
        if(std::abs(test(row, col) - ival) > 1e-12 && test(row, col) != 0)
          return false;
      } else {
        if(std::abs(test(row, col)) > 1e-12)
          return false;
      }
    }
  }

  return true;
}

BOOST_AUTO_TEST_CASE(AllTests) {
   
   BOOST_REQUIRE(test_Center());
   BOOST_REQUIRE(test_WhitenUsingEig());
   BOOST_REQUIRE(test_Orthogonalize()); 
  
}
