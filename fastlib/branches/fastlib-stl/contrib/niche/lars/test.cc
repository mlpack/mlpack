/** @file test.cc
 *
 *  Test Driver file for testing various parts of LARS
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

  bool use_cholesky = true;
  double lambda_1 = 1.0;
  double lambda_2 = 3.5;
  
  
  Lars lars;
  lars.Init(X, y, use_cholesky, lambda_1, lambda_2);

  mat Z = randu(10, 4);
  mat R = chol(trans(Z) * Z + lambda_2 * eye(4,4));

  mat ZtZ_reg = trans(Z) * Z + lambda_2 * eye(4,4);
  mat approx_ZtZ_reg = trans(R) * R;
  printf("||ZtZ_reg - approx_ZtZ_reg||_F = %e\n", norm(ZtZ_reg - approx_ZtZ_reg, "fro"));
  ZtZ_reg.print("ZtZ_reg");
  
  u32 col_to_kill = 3;
  lars.CholeskyDelete(R, col_to_kill);
  
  Z.shed_col(col_to_kill);
  mat newZ = Z;
  mat newZtnewZ_reg = trans(newZ) * newZ + lambda_2 * eye(3,3);
  mat approx_newZtnewZ_reg = trans(R) * R;
  printf("||newZtnewZ_reg - approx_newZtnewZ_reg||_F = %e\n", norm(newZtnewZ_reg - approx_newZtnewZ_reg, "fro"));
  newZtnewZ_reg.print("newZtnewZ_reg");
  
}
