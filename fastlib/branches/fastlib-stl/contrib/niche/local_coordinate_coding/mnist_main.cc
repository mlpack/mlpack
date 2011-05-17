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
  
  //const char* initial_dictionary_fullpath = 
  //  fx_param_str_req(NULL, "initial_dictionary");
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  u32 n_iterations = fx_param_int_req(NULL, "n_iterations");
  
  u32 n_atoms = fx_param_int_req(NULL, "n_atoms");
  
  
  
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
  
  // create a labels vector, NOT so that we can use it for LCC, but so we can save the labels for easy discriminative training upon exit of this program
  vec y = vec(n_points);
  y.subvec(0, n_neg_points - 1).fill(-1);
  y.subvec(n_neg_points, n_points - 1).fill(1);
  
  
  // run LCC
  LocalCoordinateCoding lcc;
  
  //mat initial_D;
  //initial_D.load("/home/niche/fastlib-stl/contrib/niche/local_coordinate_coding/D.dat");
  //initial_D.load(initial_dictionary_fullpath);
  //u32 n_atoms = initial_D.n_cols;
  
  lcc.Init(X, n_atoms, lambda);
  lcc.DataDependentRandomInitDictionary();
  
  //printf("n_atoms = %d\n", n_atoms);
  
  wall_clock timer;
  timer.tic();
  lcc.DoLCC(n_iterations);
  double n_secs = timer.toc();
  cout << "took " << n_secs << " seconds" << endl;
  
  mat learned_D;
  lcc.GetDictionary(learned_D);
  
  mat learned_V;
  lcc.GetCoding(learned_V);
  
  if(strlen(results_dir) == 0) {
    learned_D.save("D.dat", raw_ascii);
    learned_V.save("V.dat", raw_ascii);
    y.save("y.dat", raw_ascii);
  }
  else {
    char* data_fullpath = (char*) malloc(320 * sizeof(char));

    sprintf(data_fullpath, "%s/D.dat", results_dir);
    learned_D.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/V.dat", results_dir);
    learned_V.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/y.dat", results_dir);
    y.save(data_fullpath, raw_ascii);
    
    free(data_fullpath);
  }
  
  fx_done(root);
}
