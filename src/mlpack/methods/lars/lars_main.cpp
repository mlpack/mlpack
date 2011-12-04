/** @file new_main.cc
 *
 *  Driver file for testing LARS
 *
 *  @author Nishant Mehta (niche)
 */

//#include <fastlib/fastlib.h>
#include <armadillo>

#include "lars.hpp"

using namespace arma;
using namespace std;
using namespace mlpack;
using namespace mlpack::lars;

int main(int argc, char* argv[])
{
  //bool use_cholesky = false;
  double lambda_1 = 1;
  double lambda_2 = 0.5;

  u32 n = 100;
  u32 p = 10;

  mat X = randu<mat>(n,p);

  /*
  mat X_reg = zeros(n + p, p);
  X_reg(span(0, n - 1), span::all) = X;
  for(u32 i = 0; i < p; i++) {
    X_reg(n + i, i) = sqrt(lambda_2);
  }
  //X_reg.print("X_reg");
  */

  mat beta_true = zeros(p,1);
  beta_true(0) = 1;
  beta_true(1) = -1;
  beta_true(9) = 1;

  vec y = X * beta_true + 0.1 * randu<vec>(n);
  //vec y = randu(n);
  //y.load("y.dat", raw_ascii);
  //y.load("x.dat", raw_ascii);

  vec y_reg = zeros(n + p);
  y_reg.subvec(0, n - 1) = y;
  //y_reg.print("y_reg");

  mat Gram = trans(X) * X;

  LARS lars(X, y, true, lambda_1, lambda_2);
  //lars.Init(X, y, true);
  //lars.Init(X, y, false);
  //lars.SetGram(Gram.memptr(), X.n_cols);
  //lars.Init(X_reg, y_reg, false, lambda_1);
  //lars.Init(X_reg, y_reg, use_cholesky);

  lars.DoLARS();

  u32 path_length = lars.beta_path().size();

  mat beta_matrix = mat(p, path_length);
  for(u32 i = 0; i < path_length; i++)
  {
    beta_matrix.col(i) = lars.beta_path()[i];
  }
  //beta_matrix.print("beta matrix");

  vec lambda_path_vec = conv_to<colvec>::from(lars.lambda_path());
  //lambda_path_vec.print("lambda path");

  X.save("X.dat", raw_ascii);
  y.save("y.dat", raw_ascii);

  ////beta_matrix.save("beta.dat", raw_ascii);
  ////lambda_path_vec.save("lambda.dat", raw_ascii);
  vec beta;
  lars.Solution(beta);

  beta.print("final beta");
}
