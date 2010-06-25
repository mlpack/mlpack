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

#include "lin_alg.h"

using namespace arma;
using namespace linalg__private;

/***
 * Test for linalg__private::Center().  There are no edge cases here, so we'll
 * just try it once for now.
 */
bool test_center() {
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
  
  return true;
}

int main() {
  printf("test_center(): ");
  if(test_center()) {
    printf("pass\n");
    return 0;
  } else {
    printf("fail\n");
    return 1;
  }
}
