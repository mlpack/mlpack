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

using namespace arma;
using namespace std;

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);
  
  double lambda = fx_param_double_req(NULL, "lambda");
  // if using fx-run, one could just leave results_dir blank
  const char* results_dir = fx_param_str(NULL, "results_dir", "");
  
  const char* data_dir = 
    fx_param_str_req(NULL, "data_dir");
  const char* dictionary_fullpath = 
    fx_param_str_req(NULL, "dictionary");
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  
  // Load Data
  char* data_filename = (char*) malloc(320 * sizeof(char));
  sprintf(data_filename,
	  "%s/train%d.arm",
	  data_dir,
	  digit_1);
  mat X_neg;
  X_neg.load(data_filename);

  sprintf(data_filename,
	  "%s/train%d.arm",
	  data_dir,
	  digit_2);
  mat X_pos;
  X_pos.load(data_filename);
  free(data_filename);
  
  mat X = join_rows(X_neg, X_pos);
  
  u32 n_points = X.n_cols;
  
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  
  
  
  // run LCC
  
  LocalCoordinateCoding lcc;
  
  mat D;
  D.load(dictionary_fullpath);
  u32 n_atoms = D.n_cols;
  
  lcc.Init(X, n_atoms, lambda);
  lcc.SetDictionary(D);
  lcc.OptimizeCode();
  
  mat V;
  lcc.GetCoding(V);
  
  mat synthesized_X = D * V;
  
  if(strlen(results_dir) == 0) {
    V.save("V.dat", raw_ascii);
    synthesized_X.save("X_hat.dat", raw_ascii);
    X.save("X.dat", raw_ascii);
  }
  else {
    char* data_fullpath = (char*) malloc(320 * sizeof(char));

    sprintf(data_fullpath, "%s/V.dat", results_dir);
    V.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/X_hat.dat", results_dir);
    synthesized_X.save(data_fullpath, raw_ascii);

    sprintf(data_fullpath, "%s/X.dat", results_dir);
    X.save(data_fullpath, raw_ascii);
    
    free(data_fullpath);
  }
  
  fx_done(root);
}
