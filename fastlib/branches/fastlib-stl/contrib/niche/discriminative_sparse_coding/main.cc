/** @file main.cc
 *
 *  Driver file for testing LARS
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "discriminative_sparse_coding.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]) {
  DiscriminativeSparseCoding dsc;
  mat X = randu(10,100);
  vec y = randu(100);
  dsc.Init(X, y, 20, 1.0, 1.0);
}
