/** @file main.cc
 *
 *  Driver file for testing LARS
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "discr_sparse_coding.h"

using namespace arma;
using namespace std;


void Train(u32 digit,
	   u32 n_atoms, double lambda_1, double lambda_2, double lambda_w,
	   u32 n_iterations,
	   const char* initial_dictionary_filename) {
  char* file_prefix = (char*) malloc(160 * sizeof(char));
  sprintf(file_prefix, 
	  "dsc_results%dvsRest_%datoms_lambda1is%f_lambda2is%f_lambdawis%f_%diterations",
	  digit,
	  n_atoms, lambda_1, lambda_2, lambda_w, 
	  n_iterations);
  

  DiscrSparseCoding dsc;
 
  
  char* X_pos_filename = (char*) malloc(160 * sizeof(char));
  sprintf(X_pos_filename,
	  "../contrib/niche/discriminative_sparse_coding/mnist/train%d.arm",
	  digit);
  mat X_pos;
  X_pos.load(X_pos_filename);
  free(X_pos_filename);
  u32 n_pos_points = X_pos.n_cols;
  n_pos_points = (int) (n_pos_points * 0.8);
  
  mat X = zeros(X_pos.n_rows, 0);
  for(u32 cur_digit = 0; cur_digit <= 9; cur_digit++) {
    if(cur_digit == digit) {
      continue;
    }
    char* X_part_filename = (char*) malloc(160 * sizeof(char));
    sprintf(X_part_filename,
	    "../contrib/niche/discriminative_sparse_coding/mnist/train%d.arm",
	    cur_digit);
    mat X_part;
    X_part.load(X_part_filename);
    free(X_part_filename);
    u32 n_part_points = X_part.n_cols;
    n_part_points = (int) (n_part_points * 0.8);
    
    X = join_rows(X, X_part(span::all, span(0, n_part_points - 1)));
  }
  u32 n_neg_points = X.n_cols;
  
  X = join_rows(X,
		X_pos(span::all, span(0, n_pos_points - 1)));
  
  u32 n_points = X.n_cols;

  vec y = vec(n_points);
  y.subvec(0, n_neg_points - 1).fill(-1);
  y.subvec(n_neg_points, n_points - 1).fill(1);
  
  printf("%d points\n", n_points);
  
  
  
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  
  //X.save("normalized_mnist_7vs9.dat", raw_ascii);
  
  
  
  dsc.Init(X, y, n_atoms, lambda_1, lambda_2, lambda_w);
  
  double step_size = 0.1;// not used

  
  dsc.InitDictionary(initial_dictionary_filename);
  dsc.InitW();


  
  dsc.SGDOptimize(n_iterations, step_size);

  
  vec w;
  dsc.GetW(w);
  mat D;
  dsc.GetDictionary(D);

  char* w_filename = (char*) malloc(160 * sizeof(char));
  sprintf(w_filename, "%s_w.dat", file_prefix);
  w.save(w_filename, raw_ascii);
  free(w_filename);

  char* D_filename = (char*) malloc(160 * sizeof(char));
  sprintf(D_filename, "%s_D.dat", file_prefix);
  D.save(D_filename, raw_ascii);
  free(D_filename);


  mat V = mat(n_atoms, n_points);
  for(u32 i = 0; i < n_points; i++) {
    if((i % 1000) == 0) {
      printf("%i\n", i);
    }
    Lars lars;
    lars.Init(D, X.col(i), true, lambda_1, lambda_2);
    lars.DoLARS();
    vec v;
    lars.Solution(v);
    V.col(i) = v;
  }

  char* V_filename = (char*) malloc(160 * sizeof(char));
  sprintf(V_filename, "%s_V.dat", file_prefix);
  V.save(V_filename, raw_ascii);
  free(V_filename);

  char* y_filename = (char*) malloc(160 * sizeof(char));
  sprintf(y_filename, "%s_y.dat", file_prefix);
  y.save(y_filename, raw_ascii);
  free(y_filename);
  
  free(file_prefix);

  vec predictions = trans(trans(w) * V);
  vec y_hat = vec(n_points);
  for(u32 i = 0; i < n_points; i++) {
    if(predictions(i) != 0) {
      y_hat(i) =  predictions(i) / fabs(predictions(i));
    }
    else {
      y_hat(i) = 0;
    }
  }

  mat compare = join_rows(y, y_hat);
  compare.print("y y_hat");
  
  double error = 0;
  for(u32 i = 0; i < n_points; i++) {
    if(y(i) != y_hat(i)) {
      error++;
    }
  }
  error /= ((double)n_points);
  printf("error: %f\n", error);
}



void Test(u32 digit,
	  u32 n_atoms, double lambda_1, double lambda_2, double lambda_w,
	  u32 n_iterations) {
  char* file_prefix = (char*) malloc(160 * sizeof(char));
  sprintf(file_prefix, 
	  "dsc_results%dvsRest_%datoms_lambda1is%f_lambda2is%f_lambdawis%f_%diterations",
	  digit,
	  n_atoms, lambda_1, lambda_2, lambda_w, 
	  n_iterations);
  
  
  char* w_filename = (char*) malloc(160 * sizeof(char));
  sprintf(w_filename, "%s_w.dat", file_prefix);
  vec w;
  w.load(w_filename, raw_ascii);
  free(w_filename);
  
  char* D_filename = (char*) malloc(160 * sizeof(char));
  sprintf(D_filename, "%s_D.dat", file_prefix);
  mat D;
  D.load(D_filename, raw_ascii);
  free(D_filename);
  
  char* X_pos_filename = (char*) malloc(160 * sizeof(char));
  sprintf(X_pos_filename,
	  "../contrib/niche/discriminative_sparse_coding/mnist/test%d.arm",
	  digit);
  mat X_pos;
  X_pos.load(X_pos_filename);
  free(X_pos_filename);
  //u32 n_pos_points = X_pos.n_cols;
  
  mat X = zeros(X_pos.n_rows, 0);
  for(u32 cur_digit = 0; cur_digit <= 9; cur_digit++) {
    if(cur_digit == digit) {
      continue;
    }
    char* X_part_filename = (char*) malloc(160 * sizeof(char));
    sprintf(X_part_filename,
	    "../contrib/niche/discriminative_sparse_coding/mnist/test%d.arm",
	    cur_digit);
    mat X_part;
    X_part.load(X_part_filename);
    free(X_part_filename);
    
    X = join_rows(X, X_part);
  }
  u32 n_neg_points = X.n_cols;
  
  X = join_rows(X, X_pos);
  
  u32 n_points = X.n_cols;

  vec y = vec(n_points);
  y.subvec(0, n_neg_points - 1).fill(-1);
  y.subvec(n_neg_points, n_points - 1).fill(1);

  printf("%d points\n", n_points);
  
  
  
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  
  mat V = mat(n_atoms, n_points);
  for(u32 i = 0; i < n_points; i++) {
    if((i % 1000) == 0) {
      printf("%i\n", i);
    }
    Lars lars;
    lars.Init(D, X.col(i), true, lambda_1, lambda_2);
    lars.DoLARS();
    vec v;
    lars.Solution(v);
    V.col(i) = v;
  }

  char* V_filename = (char*) malloc(160 * sizeof(char));
  sprintf(V_filename, "%s_V_test.dat", file_prefix);
  V.save(V_filename, raw_ascii);
  free(V_filename);

  char* y_filename = (char*) malloc(160 * sizeof(char));
  sprintf(y_filename, "%s_y_test.dat", file_prefix);
  y.save(y_filename, raw_ascii);
  free(y_filename);

  free(file_prefix);
  
  vec predictions = trans(trans(w) * V);
  vec y_hat = vec(n_points);
  for(u32 i = 0; i < n_points; i++) {
    if(predictions(i) != 0) {
      y_hat(i) = predictions(i) / fabs(predictions(i));
    }
    else {
      y_hat(i) = 0;
    }
  }

  mat compare = join_rows(y, y_hat);
  compare.print("y y_hat");
  
  double error = 0;
  for(u32 i = 0; i < n_points; i++) {
    if(y(i) != y_hat(i)) {
      error++;
    }
  }
  error /= ((double)n_points);
  printf("error: %f\n", error);
  
}


int main(int argc, char* argv[]) {
  double lambda_1 = 0.05;
  double lambda_2 = 0.0;//0.001;
  double lambda_w = 0.1;//0.5;
  u32 n_atoms = 50;

  u32 n_LCC_iterations = 20;
  u32 n_iterations = 20000;
  
  u32 digit = 0;
  
  char* initial_directory_filename = (char*) malloc(160 * sizeof(char));
  sprintf(initial_directory_filename,
	  "LCC_results_atoms%d_lambda%f_iterations%d_D.dat",
	  n_atoms, lambda_1, n_LCC_iterations);
  
  Train(digit,
	n_atoms, lambda_1, lambda_2, lambda_w,
	n_iterations,
	initial_directory_filename);
  
  /*
  Test(digit_1, digit_2,
       n_atoms, lambda_1, lambda_2, lambda_w,
       n_iterations);
  */


}
  


