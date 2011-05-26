/** @file mnist_dlcc_main.cc
 *
 *  Driver file for testing discriminative LCC on 2 classes of MNIST
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "discr_sparse_coding.h"
#include <contrib/niche/local_coordinate_coding/lcc.h>
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


void Test(u32 digit_1, u32 digit_2,
	  u32 n_atoms, double lambda_1, double lambda_2, double lambda_w,
	  const char* data_dir,
	  const char* results_dir) {
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
  

  char* hypothesis_dir = (char*) malloc(320 * sizeof(char));
  sprintf(hypothesis_dir, 
	  "/scratch/niche/fastlib-stl/contrib/niche/discriminative_sparse_coding/fx/mnist_dlcc/mnist_from_lcc1000/digit1_%d__digit2_%d__n_iterations_1e5__n_pegasos_iterations_1e7__n_atoms_%d__lambda1_%g__lambda2_%g__lambdaw_%g",
	  digit_1, digit_2, n_atoms,
	  lambda_1, lambda_2, lambda_w);
  printf("\"%s\"\n", hypothesis_dir);
  
  sprintf(data_fullpath, "%s/D.dat", hypothesis_dir);
  mat D;
  D.load(data_fullpath);
  
  LocalCoordinateCoding lcc;
  lcc.Init(X, n_atoms, lambda_1);
  lcc.SetDictionary(D);
  lcc.OptimizeCode();
  mat V;
  lcc.GetCoding(V);
  
  sprintf(data_fullpath, "%s/w.dat", hypothesis_dir);
  vec w;
  w.load(data_fullpath);
  
  double loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "loss", loss);
  printf("Loss:\n\t %f\n", loss);
  
  
  free(hypothesis_dir);
  free(data_fullpath);
}





int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);

  
  double lambda_1 = fx_param_double_req(NULL, "lambda1");
  double lambda_2 = fx_param_double_req(NULL, "lambda2");
  double lambda_w = fx_param_double_req(NULL, "lambdaw");
  u32 n_atoms = (u32) fx_param_double_req(NULL, "n_atoms");

  /*
  //u32 n_iterations = (u32) fx_param_double(NULL, "n_iterations", 20000);
  u32 n_pegasos_iterations = 
    (u32) fx_param_double(NULL, "n_pegasos_iterations", n_iterations);
  */
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  const char* results_dir = fx_param_str(NULL, "results_dir", "");
  
  const char* data_dir = 
    fx_param_str_req(NULL, "data_dir");
  
  
  Test(digit_1, digit_2,
       n_atoms, lambda_1, lambda_2, lambda_w,
       data_dir,
       results_dir);
  
  fx_done(root);

}
  


