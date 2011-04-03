/** @file test.cc
 *
 *  Test Driver file for testing various parts of LARS
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "lars.h"

#define ERROR_TOL 1e-13


using namespace arma;
using namespace std;


bool TestElasticNet(bool use_cholesky, double lambda_1, double lambda_2) {
  mat X;
  X.load("X.dat", raw_ascii);
  
  u32 n = X.n_rows;
  u32 p = X.n_cols;
  
  vec y;
  y.load("y.dat", raw_ascii);

  Lars lars_std;
  lars_std.Init(X, y, use_cholesky, lambda_1, lambda_2);
  
  lars_std.DoLARS();
  
  

  mat X_reg = zeros(n + p, p);
  X_reg(span(0, n - 1), span::all) = X;
  for(u32 i = 0; i < p; i++) {
    X_reg(n + i, i) = sqrt(lambda_2);
  }

  vec y_reg = zeros(n + p);
  y_reg.subvec(0, n - 1) = y;

  Lars lars_explicit;
  lars_explicit.Init(X_reg, y_reg, use_cholesky, lambda_1);
  
  lars_explicit.DoLARS();
  
  

  u32 path_length_std = lars_std.beta_path().size();
  mat beta_matrix_std = mat(p, path_length_std);
  for(u32 i = 0; i < path_length_std; i++) {
    beta_matrix_std.col(i) = lars_std.beta_path()[i];
  }
  vec lambda_path_vec_std = conv_to< colvec >::from(lars_std.lambda_path());

  u32 path_length_explicit = lars_explicit.beta_path().size();
  mat beta_matrix_explicit = mat(p, path_length_explicit);
  for(u32 i = 0; i < path_length_explicit; i++) {
    beta_matrix_explicit.col(i) = lars_explicit.beta_path()[i];
  }
  vec lambda_path_vec_explicit = conv_to< colvec >::from(lars_explicit.lambda_path());

  
  double beta_error = norm(beta_matrix_std - beta_matrix_explicit, "fro");
  
  double lambda_error = norm(lambda_path_vec_std - lambda_path_vec_explicit, 2);

  if((beta_error > ERROR_TOL) || (lambda_error > ERROR_TOL)) {
    printf("beta_error = %e\n", beta_error);
    printf("lambda_error = %e\n", lambda_error);
    return true;
  }
  else {
    return false;
  }
}


int main(int argc, char* argv[]) {
  u32 n_random_tests = 1000;
  
  for(u32 i = 0; i < n_random_tests; i++) {
    double lambda_1 = drand48() * 10.0;
    double lambda_2 = drand48() * 10.0;
    for(u32 use_cholesky = 0; use_cholesky < 2; use_cholesky++) {
      if(TestElasticNet(use_cholesky, lambda_1, lambda_2)) {
	printf("Random Test %d Failed!: use_cholesky = %d, lambda_1 = %e, lambda_2 = %e\n",
	       i,
	       use_cholesky,
	       lambda_1,
	       lambda_2);
	return EXIT_FAILURE;
      }
      else {
	printf("random test %d passed: use_cholesky = %d, lambda_1 = %e, lambda_2 = %e\n",
	       i,
	       use_cholesky,
	       lambda_1,
	       lambda_2);
      }
    }
  }
  printf("All Tests Passed\n");
  return EXIT_SUCCESS;
}
