/** @file main.cc
 *
 *  Driver file for testing LCC
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "lcc.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]) {
  u32 n_atoms = 100;

  double lambda = 1.5;
  
  
  LocalCoordinateCoding lcc;

  u32 n_dims = 10;
  u32 n_points = 30;
  
  mat X = randn(n_dims, n_points);
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }

  lcc.Init(X, n_atoms, lambda);

  u32 n_iterations = 10;
  
  lcc.InitDictionary();
  
  
  
  lcc.DoLCC(n_iterations);
  
  //lcc.PrintDictionary();
  
  lcc.PrintCoding();
  
  
}
