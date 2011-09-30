/** @file mnist_dsc.cc
 *
 *  Driver file for testing Discriminative Sparse Coding on 2 classes of MNIST
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "discr_sparse_coding.h"
#include <contrib/niche/sparse_coding/sparse_coding.h>
#include <contrib/niche/pegasos/pegasos.h>

//#include <stdio.h>
//#include <unistd.h>
//#include <stdlib.h>
//#include <math.h>


using namespace arma;
using namespace std;


void SeedRandFromDevRandom() {
  FILE* random = fopen ("/dev/random", "r");
  if (random == NULL) {
    fprintf (stderr, "Cannot open /dev/random!\n");
    exit(EXIT_FAILURE);
  }

  unsigned int seed;
  fread (&seed, sizeof (seed), 1, random);
  srand(seed); /* seed the pseudo-random number generator */
}


void ShuffleColumns(mat& X) {
  // shuffle columns of each matrix
  u32 n_points = X.n_cols;
  u32 n_points_minus_1 = n_points - 1;
  u32 draw;
  vec temp;
  for(u32 i = 0; i < n_points_minus_1; i++) {
    draw = (rand() % (n_points - i)) + i;
    temp = X.col(i);
    X.col(i) = X.col(draw);
    X.col(draw) = temp;
  }
}


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


void LoadMNIST(const char* data_dir, const char* set_type, u32 digit_1, u32 digit_2,
	       mat& X_pos, mat& X_neg) {
  char* data_fullpath = (char*) malloc(320 * sizeof(char));

  sprintf(data_fullpath,
	  "%s/%s%d.arm",
	  data_dir, set_type, digit_1);
  X_neg.load(data_fullpath);
  u32 n_neg_points = X_neg.n_cols;
  printf("loaded %d negative points\n", n_neg_points);
  
  sprintf(data_fullpath,
	  "%s/%s%d.arm",
	  data_dir, set_type, digit_2);
  X_pos.load(data_fullpath);
  u32 n_pos_points = X_pos.n_cols;
  printf("loaded %d positive points\n", n_pos_points);

  // normalize each column of data
  for(u32 i = 0; i < n_neg_points; i++) {
    X_neg.col(i) /= norm(X_neg.col(i), 2);
  }
  for(u32 i = 0; i < n_pos_points; i++) {
    X_pos.col(i) /= norm(X_pos.col(i), 2);
  }
  free(data_fullpath);
}


void TrainSC(const mat& X, 
	     u32 n_atoms, double lambda, 
	     u32 n_iterations, 
	     const char* initial_dictionary_fullpath, 
	     mat& learned_D) {
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
      exit(EXIT_FAILURE);
    }
    if(initial_D.n_rows != X.n_rows) {
      fprintf(stderr, "Error: The specified initial dictionary to load has %d dimensions, but the specified data has %d dimensions! Exiting..\n",
	      initial_D.n_rows,
	      X.n_rows);
      exit(EXIT_FAILURE);
    }
    sc.SetDictionary(initial_D);
  }
  
  sc.DoSparseCoding(n_iterations);
  
  sc.GetDictionary(learned_D);
}



void TrainDSC(const mat& X, const vec& y,
	      const mat& D_reconstructive,
	      double lambda_1, double lambda_2, double lambda_w,
	      u32 n_iterations,	u32 n_pegasos_iterations,
	      mat& D_discriminative, vec& w,
	      const char* results_dir) {

  u32 n_atoms = D_reconstructive.n_cols;
  
  /// See how well initial dictionary does ///
  // for now, we assume lambda_2 = 0
  SparseCoding sc_initial;
  sc_initial.Init(X, n_atoms, lambda_1);
  sc_initial.SetDictionary(D_reconstructive);
  sc_initial.OptimizeCode();
  mat V_sc;
  sc_initial.GetCoding(V_sc);
  
  // now that we have a coding, run Pegasos to optimize w
  Pegasos pegasos_sc;
  printf("n_pegasos_iterations = %d\n", n_pegasos_iterations);
  pegasos_sc.Init(V_sc, y, lambda_w, n_pegasos_iterations);
  pegasos_sc.DoPegasos();
  vec w_sc = pegasos_sc.GetW();

  // store some results for sparse coding + pegasos
  double sc_loss = ComputeLoss(V_sc, y, w_sc);
  fx_result_double(NULL, "train_sc_loss", sc_loss);
  printf("SC Pegasos Loss:\n\t %f\n", sc_loss);
  
  // start discriminative sparse coding
  DiscrSparseCoding dsc;
  dsc.Init(X, y, n_atoms, lambda_1, lambda_2, lambda_w);
  dsc.SetDictionary(D_reconstructive);
  dsc.SetW(w_sc); // initialize w by using the solution to an SVM problem from the coding to the original dictionary
  printf("about to call sgdoptimize for %d iterations\n", n_iterations);
  dsc.SGDOptimize(n_iterations);
  
  dsc.GetW(w);
  dsc.GetDictionary(D_discriminative);
  
  // now that we've learned a dictionary, do a final coding step of SC
  // for now, we assume lambda_2 = 0
  SparseCoding sc_final;
  sc_final.Init(X, n_atoms, lambda_1);
  sc_final.SetDictionary(D_discriminative);
  sc_final.OptimizeCode();
  mat V;
  sc_final.GetCoding(V);

  // store some resuls for discriminative sparse coding
  double dsc_pre_pegasos_loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "train_dsc_pre_pegasos_loss", dsc_pre_pegasos_loss);
  printf("DSC Pre-Pegasos Loss:\n\t%f\n", dsc_pre_pegasos_loss);
  
  // now that we have a coding, run Pegasos to optimize w
  Pegasos pegasos;
  printf("n_pegasos_iterations = %d\n",
	 n_pegasos_iterations);
  pegasos.Init(V, y, lambda_w, n_pegasos_iterations);
  pegasos.DoPegasos();
  w = pegasos.GetW();

  // store some results for discriminative sparse coding + pegasos
  double dsc_loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "train_dsc_loss", dsc_loss);
  printf("DSC Loss: %f\n", dsc_loss);
  
  // save various results files
  if(strlen(results_dir) == 0) {
    w.save("w.dat", raw_ascii);
    D_reconstructive.save("D_sc.dat", raw_ascii);
    D_discriminative.save("D_dsc.dat", raw_ascii);
    V.save("V.dat", raw_ascii);
    y.save("y.dat", raw_ascii);
  }
  else {
    char* data_fullpath = (char*) malloc(320 * sizeof(char));
    sprintf(data_fullpath, "%s/w.dat", results_dir);
    w.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/D_sc.dat", results_dir);
    D_reconstructive.save(data_fullpath, raw_ascii);

    sprintf(data_fullpath, "%s/D_dsc.dat", results_dir);
    D_discriminative.save(data_fullpath, raw_ascii);

    sprintf(data_fullpath, "%s/V.dat", results_dir);
    V.save(data_fullpath, raw_ascii);
    
    sprintf(data_fullpath, "%s/y.dat", results_dir);
    y.save(data_fullpath, raw_ascii);
    free(data_fullpath);
  }
}


void Test(const mat& X, const vec& y,
	  const mat& D, const vec& w,
	  double lambda_1) {
  u32 n_atoms = D.n_cols;
  
  SparseCoding sc;
  sc.Init(X, n_atoms, lambda_1);
  sc.SetDictionary(D);
  sc.OptimizeCode();
  mat V;
  sc.GetCoding(V);
  
  double loss = ComputeLoss(V, y, w);
  fx_result_double(NULL, "test_dsc_loss", loss);
  printf("Test Loss:\n\t %f\n", loss);
}


int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);
  
  SeedRandFromDevRandom();

  double lambda_1 = fx_param_double_req(NULL, "lambda1");
  double lambda_2 = fx_param_double_req(NULL, "lambda2");
  double lambda_w = fx_param_double_req(NULL, "lambdaw");
  u32 n_atoms = (u32) fx_param_double_req(NULL, "n_atoms");

  u32 n_sc_iterations = (u32) fx_param_double_req(NULL, "n_sc_iterations");
  u32 n_dsc_iterations = (u32) fx_param_double_req(NULL, "n_dsc_iterations");
  u32 n_pegasos_iterations = 
    (u32) fx_param_double_req(NULL, "n_pegasos_iterations");
  
  u32 digit_1 = fx_param_int_req(NULL, "digit1");
  u32 digit_2 = fx_param_int_req(NULL, "digit2");
  
  const char* initial_dictionary_fullpath = 
    fx_param_str(NULL, "initial_dictionary", "");
  
  const char* results_dir = fx_param_str(NULL, "results_dir", "");
  
  const char* data_dir = 
    fx_param_str_req(NULL, "data_dir");
  

  // Load MNIST training set for both classes
  mat X_pos_train, X_neg_train, X_pos_test, X_neg_test;
  LoadMNIST(data_dir, "train",
	    digit_1, digit_2,
	    X_pos_train, X_neg_train);
  LoadMNIST(data_dir, "test",
	    digit_1, digit_2,
	    X_pos_test, X_neg_test);

  mat X_neg = join_rows(X_neg_train, X_neg_test);
  mat X_pos = join_rows(X_pos_train, X_pos_test);

  // shuffle columns of each dataset
  ShuffleColumns(X_neg);
  ShuffleColumns(X_pos);
  
  u32 neg_cut_ind = 400 - 1; //(X_neg.n_cols / 2) - 1;
  u32 pos_cut_ind = 400 - 1; //(X_pos.n_cols / 2) - 1;
  
  

  mat X_train = join_rows(X_neg(span::all, span(0, neg_cut_ind)), 
			  X_pos(span::all, span(0, pos_cut_ind)));
  u32 n_neg_train = neg_cut_ind + 1;
  u32 n_pos_train = pos_cut_ind + 1;
  u32 n_train = n_neg_train + n_pos_train;
  vec y_train = vec(n_train);
  y_train.subvec(0, n_neg_train - 1).fill(-1);
  y_train.subvec(n_neg_train, n_train - 1).fill(1);

  mat X_test = join_rows(X_neg(span::all, span(neg_cut_ind + 1, X_neg.n_cols - 1)),
			 X_pos(span::all, span(pos_cut_ind + 1, X_pos.n_cols - 1)));
  u32 n_neg_test = X_neg.n_cols - n_neg_train;
  u32 n_pos_test = X_pos.n_cols - n_pos_train;
  u32 n_test = n_neg_test + n_pos_test;
  vec y_test = vec(n_test);
  y_test.subvec(0, n_neg_test - 1).fill(-1);
  y_test.subvec(n_neg_test, n_test - 1).fill(1);

  
  printf("X_train is %d by %d\n", X_train.n_rows, X_train.n_cols);
  printf("X_test is %d by %d\n", X_test.n_rows, X_test.n_cols);
  
  mat D_reconstructive;
  TrainSC(X_train, 
	  n_atoms, lambda_1, 
	  n_sc_iterations, 
	  initial_dictionary_fullpath, 
	  D_reconstructive);
  
  mat D_discriminative;
  vec w;
  TrainDSC(X_train, y_train,
	   D_reconstructive,
	   lambda_1, lambda_2, lambda_w,
	   n_dsc_iterations, n_pegasos_iterations,
	   D_discriminative, w,
	   results_dir);
  
  Test(X_test, y_test,
       D_discriminative, w,
       lambda_1);
  
  fx_done(root);

}
  


