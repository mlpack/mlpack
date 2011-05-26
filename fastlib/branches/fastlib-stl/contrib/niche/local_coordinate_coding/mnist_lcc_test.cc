/** @file main.cc
 *
 *  Driver file for testing LCC
 *
 *  @author Nishant Mehta (niche)
 */
// example for running program via fx-run on no.cc.gatech.edu:
//   fx-run mnist_lcc /scratch/niche/fastlib-stl/build/bin/mnist_lcc --lambda=0.05, --data_dir=/scratch/niche/fastlib-stl/contrib/niche/discriminative_sparse_coding/mnist --digit1=4, --digit2=9, --n_iterations=1, --n_atoms=50,

#include <fastlib/fastlib.h>
#include <armadillo>

#include "lcc.h"
#include <contrib/niche/pegasos/pegasos.h>

using namespace arma;
using namespace std;


double ComputeLoss(const mat& X, const vec&y, const vec&w) {
  u32 n_points = y.n_elem;
  vec predictions = trans(trans(w) * X);
  vec y_hat = vec(n_points);
  for(u32 i = 0; i < n_points; i++) {
    if(predictions(i) != 0) {
      y_hat(i) =  predictions(i) / fabs(predictions(i));
    }
    else {
      y_hat(i) = 0;
    }
  }
  
  double loss = 0;
  for(u32 i = 0; i < n_points; i++) {
    if(y(i) != y_hat(i)) {
      loss++;
    }
  }
  loss /= ((double)n_points);
  return loss;
}


void LearnWAndGetD(u32 digit_1, u32 digit_2,
		   u32 n_atoms, double lambda, double lambda_w,
		   const char* data_dir,
		   vec& w, mat& D) {
  
  // Load Data
  char* data_filename = (char*) malloc(320 * sizeof(char));
  sprintf(data_filename,
	  "%s/train%d.arm",
	  data_dir,
	  digit_1);
  mat X_neg;
  X_neg.load(data_filename);
  u32 n_neg_points = X_neg.n_cols;
  
  sprintf(data_filename,
	  "%s/train%d.arm",
	  data_dir,
	  digit_2);
  mat X_pos;
  X_pos.load(data_filename);
  u32 n_pos_points = X_pos.n_cols;
  free(data_filename);
  
  mat X = join_rows(X_neg, X_pos);
  u32 n_points = X.n_cols;
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  

  char* dictionary_fullpath = (char*) malloc(320 * sizeof(char));
  sprintf(dictionary_fullpath, 
	  "/scratch/niche/fastlib-stl/contrib/niche/local_coordinate_coding/fx/mnist_lcc/mnist/digit1_%d__digit2_%d__n_iterations_100__n_atoms_%d__lambda_%g/D.dat",
	  digit_1, digit_2, n_atoms,
	  lambda);
  printf("\"%s\"\n", dictionary_fullpath);
  
  LocalCoordinateCoding lcc;
  lcc.Init(X, n_atoms, lambda);
  D.load(dictionary_fullpath);
  lcc.SetDictionary(D);
  lcc.OptimizeCode();
  mat V;
  lcc.GetCoding(V);
  
  vec y = vec(n_points);
  y.subvec(0, n_neg_points - 1).fill(-1);
  y.subvec(n_neg_points, n_points - 1).fill(1);
    
  
  u32 n_pegasos_iterations = 10000000;
  Pegasos pegasos;
  pegasos.Init(V, y, lambda_w, n_pegasos_iterations);
  pegasos.DoPegasos();
  w = pegasos.GetW(); // use valgrind and also ask Ryan if this is safe
}


void Test(u32 digit_1, u32 digit_2,
	  u32 n_atoms, double lambda,
	  const char* data_dir,
	  const vec& w, const mat& D) {
  
  char* data_fullpath = (char*) malloc(320 * sizeof(char));

  sprintf(data_fullpath,
	  "%s/test%d.arm",
	  data_dir, digit_1);
  mat X_neg;
  X_neg.load(data_fullpath);
  u32 n_neg_points = X_neg.n_cols;
  printf("%d negative points\n", n_neg_points);
  
  sprintf(data_fullpath,
	  "%s/test%d.arm",
	  data_dir, digit_2);
  mat X_pos;
  X_pos.load(data_fullpath);
  u32 n_pos_points = X_pos.n_cols;
  printf("%d positive points\n", n_pos_points);

  mat X = join_rows(X_neg, X_pos);
  u32 n_points = X.n_cols;
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  
  vec y = vec(n_points);
  y.subvec(0, n_neg_points - 1).fill(-1);
  y.subvec(n_neg_points, n_points - 1).fill(1);
  
  printf("%d points\n", n_points);
  

  LocalCoordinateCoding lcc;
  lcc.Init(X, n_atoms, lambda);
  lcc.SetDictionary(D);
  lcc.OptimizeCode();
  mat V;
  lcc.GetCoding(V);
  
  double loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "loss", loss);
  printf("Loss:\n\t %f\n", loss);
  
  
  free(data_fullpath);

}


int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  double lambda_w = fx_param_double_req(NULL, "lambdaw");
  
  const char* data_dir = 
    fx_param_str_req(NULL, "data_dir");
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  //u32 n_iterations = (u32) fx_param_double_req(NULL, "n_iterations");
  
  u32 n_atoms = (u32) fx_param_double_req(NULL, "n_atoms");

  vec w;
  mat D;
  LearnWAndGetD(digit_1, digit_2,
		n_atoms, lambda, lambda_w,
		data_dir,
		w, D);
  
  Test(digit_1, digit_2,
       n_atoms, lambda,
       data_dir,
       w, D);

  
  fx_done(root);
}
