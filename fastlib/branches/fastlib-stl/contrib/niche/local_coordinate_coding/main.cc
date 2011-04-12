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

  double lambda = 0.05;
  
  
  LocalCoordinateCoding lcc;
  
  /*
  u32 n_atoms = 100;

  u32 n_dims = 10;
  u32 n_points = 30;

  mat X = randn(n_dims, n_points);
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  */
  mat X;
  X.load("/home/niche/fastlib-stl/contrib/niche/local_coordinate_coding/X.dat");

  mat D;
  D.load("/home/niche/fastlib-stl/contrib/niche/local_coordinate_coding/D.dat");
  
  u32 n_atoms = D.n_cols;
  
  lcc.Init(X, n_atoms, lambda);
  
  lcc.SetDictionary(D);
  
  printf("n_atoms = %d\n", n_atoms);
  
  lcc.DoLCC(1);
  //lcc.PrintDictionary();
  mat Dic;
  lcc.GetDictionary(Dic);
  Dic.save("Dic.dat", raw_ascii);
  
  
  /*
  
  u32 n_iterations = 10;

  lcc.InitDictionary();
  
  lcc.DoLCC(n_iterations);
  
  */
  
}
