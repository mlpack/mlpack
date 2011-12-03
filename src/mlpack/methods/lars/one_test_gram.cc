/** @file test.cc
 *
 *  Test Driver file for testing various parts of LARS
 *
 *  @author Nishant Mehta (niche)
 */

#include <armadillo>

#include "lars.h"

#define ERROR_TOL 1e-13


using namespace arma;
using namespace std;



int main(int argc, char* argv[]) {
  double lambda_1 = 0;
  double lambda_2 = 1;
  
  mat X;
  X.load("X.dat", raw_ascii);
  
  vec y;
  y.load("y.dat", raw_ascii);

  Lars lars;
  lars.Init(X, y, false, lambda_1, lambda_2);
  
  lars.DoLARS();
  
  u32 path_length = lars.beta_path().size();
  /*
  u32 n = X.n_rows;
  u32 p = X.n_cols;
  
  mat beta_matrix = mat(p, path_length);
  for(u32 i = 0; i < path_length; i++) {
    beta_matrix.col(i) = lars.beta_path()[i];
  }
  beta_matrix.print("beta matrix");
  
  vec lambda_path_vec = conv_to< colvec >::from(lars.lambda_path());
  lambda_path_vec.print("lambda path");
  */
  
  
  vec beta;
  lars.Solution(beta);
  beta.print("final beta");
  
  printf(" path length = %d\n", path_length);
  
  
  double temp = norm(X * beta - y, "fro");
  double obj_val = 0.5 * temp * temp + lambda_1 * norm(beta, 1);
  
  printf("objective value = %e\n", obj_val);
  
  return EXIT_SUCCESS;
}
