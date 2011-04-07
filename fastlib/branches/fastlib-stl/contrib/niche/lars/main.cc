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
  
  bool use_cholesky = true;
  double lambda_1 = 1e-5;//1.0;//1e-5;//2.0;//0.001;//0.12;
  double lambda_2 = 1.0;
  
  //u32 n = 100;
  //u32 p = 50;

  //std::srand(17);
  mat X;// = randu<mat>(n,p);
  X.load("X.dat", raw_ascii);
  
  u32 n = X.n_rows;
  u32 p = X.n_cols;
  
  mat X_reg = zeros(n + p, p);
  X_reg(span(0, n - 1), span::all) = X;
  for(u32 i = 0; i < p; i++) {
    X_reg(n + i, i) = sqrt(lambda_2);
  }
  //X_reg.print("X_reg");
  
  /*
  mat beta = zeros(p,1);
  beta(0) = 1;
  beta(1) = -1;
  beta(9) = 1;
  */
  
  vec y;// = X * beta + 0.1 * randu<vec>(n);
  //vec y = randu(n);
  y.load("y.dat", raw_ascii);
  
  vec y_reg = zeros(n + p);
  y_reg.subvec(0, n - 1) = y;
  //y_reg.print("y_reg");
  
  
  Lars lars;
  lars.Init(X, y, use_cholesky, lambda_1, lambda_2);
  //lars.Init(X_reg, y_reg, use_cholesky, lambda_1);
  
  lars.DoLARS();
  
  u32 path_length = lars.beta_path().size();
  
  mat beta_matrix = mat(p, path_length);
  for(u32 i = 0; i < path_length; i++) {
    beta_matrix.col(i) = lars.beta_path()[i];
  }
  //beta_matrix.print("beta matrix");

  vec lambda_path_vec = conv_to< colvec >::from(lars.lambda_path());
  //lambda_path_vec.print("lambda path");
  
  
  //X.save("X.dat", raw_ascii);
  //y.save("y.dat", raw_ascii);
  beta_matrix.save("beta.dat", raw_ascii);
  lambda_path_vec.save("lambda.dat", raw_ascii);
  
  
  vec beta;
  lars.Solution(beta);
  
  beta.print("final beta");
}
