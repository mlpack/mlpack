/** @file main.cc
 *
 *  Driver file for testing LCC
 *
 *  @author Nishant Mehta (niche)
 */

#include <fastlib/fastlib.h>
#include <armadillo>

#include "lcc.h"

using namespace arma;
using namespace std;

int main(int argc, char* argv[]) {
  fx_module* root = fx_init(argc, argv, NULL);
  
  double lambda = fx_param_double_req(NULL, "lambda"); // 0.05
  // if using fx-run, one could just leave results_dir blank
  const char* results_dir = fx_param_str(NULL, "results_dir", "");
  
  const char* data_fullpath = 
    fx_param_str_req(NULL, "data_fullpath");
  const char* initial_dictionary_fullpath = 
    fx_param_str_req(NULL, "initial_dictionary");

  u32 n_iterations = fx_param_int_req(NULL, "n_iterations");

  
  LocalCoordinateCoding lcc;
  
  mat X;
  //X.load("/home/niche/fastlib-stl/contrib/niche/local_coordinate_coding/X.dat");
  X.load(data_fullpath);

  mat initial_D;
  //initial_D.load("/home/niche/fastlib-stl/contrib/niche/local_coordinate_coding/D.dat");
  initial_D.load(initial_dictionary_fullpath);
  u32 n_atoms = initial_D.n_cols;
  
  lcc.Init(X, n_atoms, lambda);
  lcc.SetDictionary(initial_D);
  
  //printf("n_atoms = %d\n", n_atoms);
  
  lcc.DoLCC(n_iterations);

  mat learned_D;
  lcc.GetDictionary(learned_D);

  if(strlen(results_dir) == 0) {
    learned_D.save("D.dat", raw_ascii);
  }
  else {
    char* learned_dictionary_fullpath = (char*) malloc(320 * sizeof(char));
    sprintf(learned_dictionary_fullpath,
	    "%s/D.dat",
	    results_dir);
    learned_D.save(learned_dictionary_fullpath, raw_ascii);
    free(learned_dictionary_fullpath);
  }
  
  fx_done(root);
}
