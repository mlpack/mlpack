/** @file main.cc
 *
 *  Driver file for testing Sparse Coding
 *
 *  @author Nishant Mehta (niche)
 */
// example for running program via fx-run on no.cc.gatech.edu:
//   fx-run mnist_sc /scratch/niche/fastlib-stl/build/bin/mnist_sc --lambda=0.05, --data_dir=/scratch/niche/fastlib-stl/contrib/niche/discriminative_sparse_coding/mnist --digit1=4, --digit2=9, --n_iterations=1, --n_atoms=50,

#include <fastlib/fastlib.h>
#include <armadillo>

#include "sparse_coding.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);
  
  std::srand(time(NULL));
  
  double lambda = fx_param_double_req(NULL, "lambda");
  
  // if using fx-run, one could just leave results_dir blank
  const char* results_dir = fx_param_str(NULL, "results_dir", "");
  
  const char* data_dir = 
    fx_param_str_req(NULL, "data_dir");
  
  const char* initial_dictionary_fullpath = 
    fx_param_str(NULL, "initial_dictionary", "");
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  u32 n_iterations = (u32) fx_param_double_req(NULL, "n_iterations");
  
  u32 n_atoms = (u32) fx_param_double_req(NULL, "n_atoms");
  
  
  
  // Load Data
  char* data_filename = (char*) malloc(320 * sizeof(char));
  sprintf(data_filename,
	  "%s/train%d.arm",
	  data_dir,
	  digit_1);
  mat X_neg;
  X_neg.load(data_filename);
  // SHRINK DATASET SIZES FOR QUICK EXPERIMENTS ON NEWTON'S METHOD IN THE DUAL
  X_neg = X_neg(span::all, span(0, 199));
  u32 n_neg_points = X_neg.n_cols;
  
  sprintf(data_filename,
	  "%s/train%d.arm",
	  data_dir,
	  digit_2);
  mat X_pos;
  X_pos.load(data_filename);
  // SHRINK DATASET SIZES FOR QUICK EXPERIMENTS ON NEWTON'S METHOD IN THE DUAL
  X_pos = X_pos(span::all, span(0, 199));
  u32 n_pos_points = X_pos.n_cols;
  free(data_filename);
  
  
  mat X = join_rows(X_neg, X_pos);
  u32 n_points = X.n_cols;
  // normalize each column of data
  for(u32 i = 0; i < n_points; i++) {
    X.col(i) /= norm(X.col(i), 2);
  }
  
  // create a labels vector, NOT so that we can use it for Sparse Coding, but so we can save the labels for easy discriminative training upon exit of this program
  vec y = vec(n_points);
  y.subvec(0, n_neg_points - 1).fill(-1);
  y.subvec(n_neg_points, n_points - 1).fill(1);
  
  
  // run Sparse Coding
  SparseCoding sc;
  
  sc.Init(X, n_atoms, lambda);
  if(strlen(initial_dictionary_fullpath) == 0) {
    sc.DataDependentRandomInitDictionary();
    //sc.RandomInitDictionary();
  }
  else {
    mat initial_D;
    initial_D.load(initial_dictionary_fullpath);
    if(initial_D.n_cols != n_atoms) {
      fprintf(stderr, "Error: The specified initial dictionary to load has %d atoms, but the learned dictionary was specified to have %d atoms! Exiting..\n",
	      initial_D.n_cols,
	      n_atoms);
      return EXIT_FAILURE;
    }
    if(initial_D.n_rows != X.n_rows) {
      fprintf(stderr, "Error: The specified initial dictionary to load has %d dimensions, but the specified data has %d dimensions! Exiting..\n",
	      initial_D.n_rows,
	      X.n_rows);
      return EXIT_FAILURE;
    }
    sc.SetDictionary(initial_D);
  }
  
  wall_clock timer;
  timer.tic();
  sc.DoSparseCoding(n_iterations);
  double n_secs = timer.toc();
  cout << "took " << n_secs << " seconds" << endl;
  
  mat learned_D;
  sc.GetDictionary(learned_D);
  
  mat learned_V;
  sc.GetCoding(learned_V);
  
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
