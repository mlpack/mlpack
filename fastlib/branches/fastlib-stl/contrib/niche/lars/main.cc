/** @file main.cc
 *
 *  Driver file for testing LARS
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "lars.h"

using namespace arma;

int main(int argc, char* argv[]) {

  Lars lars;

  std::srand(17);
  mat X = randu<mat>(100,10);
  mat beta = zeros(10,1);
  beta(0) = 1;
  beta(1) = 1;
  beta(9) = 1;
  vec y = X * beta;

  lars.Init(X, y);
 
  
  lars.DoLARS();
  
}
