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
  // [[2 4 6 8 10]]
  // so result should be
  // [[-2 -4 -6 -8 -10]
  //  [-1 -2 -3 -4 -5 ]
  //  [ 0  0  0  0  0 ]
  //  [ 1  2  3  4  5 ]
  //  [ 2  4  6  8  10]]
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 5; col++) {
      if(tmp_out(row, col) != (row * (col + 1)) - ((col + 1) * 2))
        return false;
    }
  }
  
  mat tmp2(5, 6);
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 6; col++)
      tmp2(row, col) = row * (col + 1);
  }
  Center(tmp2, tmp_out);

  // average should be
  // [[2 4 6 8 10 12]]
  // so result should be
  // [[-2 -4 -6 -8 -10 -12]
  //  [-1 -2 -3 -4 -5  -6 ]
  //  [ 0  0  0  0  0   0 ]
  //  [ 1  2  3  4  5   6 ]
  //  [ 2  4  6  8  10  12]]
  for(int row = 0; row < 5; row++) {
    for(int col = 0; col < 6; col++) {
      if(tmp_out(row, col) != (row * (col + 1)) - ((col + 1) * 2))
        return false;
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
  for(int row = 0; row < test.n_rows; row++) {
    for(int col = 0; col < test.n_cols; col++) {
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

int main() {
  printf("test_Center(): ");
  if(test_Center()) {
    printf("pass\n");
  } else {
    printf("fail\n");
    return 1;
  }
  
  printf("test_WhitenUsingEig(): ");
  if(test_WhitenUsingEig()) {
    printf("pass\n");
  } else {
    printf("fail\n");
    return 1;
  }
  
  printf("test_Orthogonalize(): ");
  if(test_Orthogonalize()) {
    printf("pass\n");
  } else {
    printf("fail\n");
    return 1;
  }

  return 0;
}
