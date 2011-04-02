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
using namespace std;

int main(int argc, char* argv[]) {
  
  int n = 100;
  int p = 10;

  std::srand(17);
  mat X = randu<mat>(n,p);
  X.load("X.dat", raw_ascii);

  mat beta = zeros(p,1);
  beta(0) = 1;
  beta(1) = -1;
  beta(9) = 1;

  vec y = X * beta + 0.1 * randu<vec>(n);
  y.load("y.dat", raw_ascii);

  bool use_cholesky = !true;
  double lambda_1 = 0.12;//1.0;
  
  Lars lars;
  lars.Init(X, y, use_cholesky, lambda_1);
  
  lars.DoLARS();
  
  u32 path_length = lars.beta_path().size();
  
  mat beta_matrix = mat(p, path_length);
  for(u32 i = 0; i < path_length; i++) {
    beta_matrix.col(i) = lars.beta_path()[i];
  }
  //beta_matrix.print("beta matrix");

  vec lambda_path_vec = conv_to< colvec >::from(lars.lambda_path());
  lambda_path_vec.print("lambda path");
  
  X.save("X.dat", raw_ascii);
  y.save("y.dat", raw_ascii);
  beta_matrix.save("beta.dat", raw_ascii);
  lambda_path_vec.save("lambda.dat", raw_ascii);
}
