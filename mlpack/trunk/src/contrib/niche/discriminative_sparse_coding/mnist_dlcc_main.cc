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


void Train(u32 digit_1, u32 digit_2,
	   u32 n_atoms, double lambda_1, double lambda_2, double lambda_w,
	   u32 n_iterations,
	   u32 n_pegasos_iterations,
	   const char* initial_dictionary_fullpath,
	   const char* data_dir,
	   const char* results_dir) {
  DiscrSparseCoding dsc;
  
  char* data_fullpath = (char*) malloc(320 * sizeof(char));

  sprintf(data_fullpath,
	  "%s/train%d.arm",
	  data_dir, digit_1);
  mat X_neg;
  X_neg.load(data_fullpath);
  u32 n_neg_points = X_neg.n_cols;
  printf("%d negative points\n", n_neg_points);
  
  sprintf(data_fullpath,
	  "%s/train%d.arm",
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
  
  
  dsc.Init(X, y, n_atoms, lambda_1, lambda_2, lambda_w);
  
  double step_size = 0.1;// not used
  
  dsc.InitDictionary(initial_dictionary_fullpath);

  
    
  /// See how well initial dictionary does ///
  // now that we've learned a dictionary, do a final coding step of LCC
  // for now, we assume lambda_2 = 0
  LocalCoordinateCoding lcc_initial;
  lcc_initial.Init(X, n_atoms, lambda_1);
  mat D_initial;
  dsc.GetDictionary(D_initial);
  lcc_initial.SetDictionary(D_initial);
  lcc_initial.OptimizeCode();
  mat V_lcc;
  lcc_initial.GetCoding(V_lcc);
  
  
  // now that we have a coding, run Pegasos to optimize w
  Pegasos pegasos_lcc;
  printf("n_pegasos_iterations = %d\n",
	 n_pegasos_iterations);
  pegasos_lcc.Init(V_lcc, y, lambda_w, n_pegasos_iterations);
  pegasos_lcc.DoPegasos();
  vec w_lcc = pegasos_lcc.GetW();
  
  double lcc_loss = ComputeLoss(V_lcc, y, w_lcc);
  fx_result_double(NULL, "lcc_loss", lcc_loss);
  printf("LCC Pegasos Loss:\n\t %f\n", lcc_loss);
  
  
  
  
  //dsc.InitW();
  dsc.SetW(w_lcc); // initialize w by using the solution to an SVM problem from the coding to the original dictionary
  
  
  dsc.SGDOptimize(n_iterations, step_size);
  
  vec w;
  dsc.GetW(w);
  mat D;
  dsc.GetDictionary(D);
  
  
  // now that we've learned a dictionary, do a final coding step of LCC
  // for now, we assume lambda_2 = 0
  LocalCoordinateCoding lcc_final;
  lcc_final.Init(X, n_atoms, lambda_1);
  lcc_final.SetDictionary(D);
  lcc_final.OptimizeCode();
  mat V;
  lcc_final.GetCoding(V);

  double dlcc_pre_pegasos_loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "dlcc_pre_pegasos_loss", dlcc_pre_pegasos_loss);
  printf("DLCC Pre-Pegasos Loss:\n\t%f\n", dlcc_pre_pegasos_loss);
  
  
  // now that we have a coding, run Pegasos to optimize w
  Pegasos pegasos;
  printf("n_pegasos_iterations = %d\n",
	 n_pegasos_iterations);
  pegasos.Init(V, y, lambda_w, n_pegasos_iterations);
  pegasos.DoPegasos();
  w = pegasos.GetW();

  double dlcc_loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "dlcc_loss", dlcc_loss);
  printf("DLCC Loss: %f\n", dlcc_loss);
  
  
  
  //mat synthesized_X = D * V;
  
  if(strlen(results_dir) == 0) {
    w.save("w.dat", raw_ascii);
    D.save("D.dat", raw_ascii);
    V.save("V.dat", raw_ascii);
    y.save("y.dat", raw_ascii);
  }
  else {
    sprintf(data_fullpath, "%s/w.dat", results_dir);
    w.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/D.dat", results_dir);
    D.save(data_fullpath, raw_ascii);

    sprintf(data_fullpath, "%s/V.dat", results_dir);
    V.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/y.dat", results_dir);
    y.save(data_fullpath, raw_ascii);
  }
  
  free(data_fullpath);
}





int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);

  double lambda_1 = fx_param_double_req(NULL, "lambda1");
  double lambda_2 = fx_param_double_req(NULL, "lambda2");
  double lambda_w = fx_param_double_req(NULL, "lambdaw");
  u32 n_atoms = (u32) fx_param_double_req(NULL, "n_atoms");

  u32 n_iterations = (u32) fx_param_double(NULL, "n_iterations", 20000);
  u32 n_pegasos_iterations = 
    (u32) fx_param_double(NULL, "n_pegasos_iterations", n_iterations);
  
  //u32 n_LCC_iterations = 30;
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  const char* initial_dictionary_fullpath = 
    fx_param_str_req(NULL, "initial_dictionary");
  
  const char* results_dir = fx_param_str(NULL, "results_dir", "");
  
  const char* data_dir = 
    fx_param_str_req(NULL, "data_dir");
  
  
  Train(digit_1, digit_2,
	n_atoms, lambda_1, lambda_2, lambda_w,
	n_iterations,
	n_pegasos_iterations,
	initial_dictionary_fullpath,
	data_dir,
	results_dir);
  

  fx_done(root);

}
  


