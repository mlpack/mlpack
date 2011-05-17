/** @file mnist_dlcc_main.cc
 *
 *  Driver file for testing discriminative LCC on 2 classes of MNIST
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include <mlpack/svm/svm.h>
//#include <fastlib/math/statistics.h>

//#include <fastlib/base/arma_compat.h>

// if these are included first, we have compilation errors from svm.h. why??
#include "discr_sparse_coding.h"
#include <contrib/niche/local_coordinate_coding/lcc.h>



using namespace arma;
using namespace std;




void Train(u32 digit_1, u32 digit_2,
	   u32 n_atoms, double lambda_1, double lambda_2, double lambda_w,
	   u32 n_iterations,
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
  dsc.InitW(); // we should initialize w by using the solution to an SVM problem from the coding to the original dictionary
  
  
  dsc.SGDOptimize(n_iterations, step_size);
  
  vec w;
  dsc.GetW(w);
  mat D;
  dsc.GetDictionary(D);
  
  
  // now that we've learned a dictionary, do a final coding step of LCC
  // for now, we assume lambda_2 = 0
  LocalCoordinateCoding lcc;
  lcc.Init(X, n_atoms, lambda_1);
  lcc.SetDictionary(D);
  lcc.OptimizeCode();
  mat V;
  lcc.GetCoding(V);
  
  //mat synthesized_X = D * V;
  
  if(strlen(results_dir) == 0) {
    w.save("w.dat", raw_ascii);
    D.save("D.dat", raw_ascii);
    V.save("V.dat", raw_ascii);
    //synthesized_X.save("X_hat.dat", raw_ascii);
    y.save("y.dat", raw_ascii);
  }
  else {
    sprintf(data_fullpath, "%s/w.dat", results_dir);
    w.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/D.dat", results_dir);
    D.save(data_fullpath, raw_ascii);

    sprintf(data_fullpath, "%s/V.dat", results_dir);
    V.save(data_fullpath, raw_ascii);
    
    //sprintf(data_fullpath, "%s/X_hat.dat", results_dir);
    //synthesized_X.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/y.dat", results_dir);
    y.save(data_fullpath, raw_ascii);
  }
  
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

  free(data_fullpath);



  /*
  // given the coding V and labels y, learn an SVM use Hua's code
  SVM<SVMLinearKernel> svm;
  
  u32 learner_typeid = 0; // corresponds to support vector classification
  Dataset trainset;
  mat svm_data(V.n_rows + 1, V.n_cols);
  svm_data(span(0, V.n_rows - 1), span::all) = V;
  svm_data(V.n_rows, span::all) = trans(y);
  trainset.CopyMatrix(svm_data);
  
  datanode* svm_module = fx_submodule(fx_root, "svm");
  fx_set_param_double(svm_module, "c", lambda_w);
  
  svm.InitTrain(learner_typeid, trainset, svm_module);
  */  


}



int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);

  double lambda_1 = fx_param_double_req(NULL, "lambda1");
  double lambda_2 = fx_param_double_req(NULL, "lambda2");
  double lambda_w = fx_param_double_req(NULL, "lambdaw");
  u32 n_atoms = fx_param_int_req(NULL, "n_atoms");

  u32 n_iterations = fx_param_int(NULL, "n_iterations", 20000);
  
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
	initial_dictionary_fullpath,
	data_dir,
	results_dir);
  

  fx_done(root);

}
  


