#include "kde_cv.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec.
  fx_init(argc, argv, NULL);

  // Initialize the random number seed.
  srand(time(NULL));

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode* kde_module = fx_submodule(fx_root, "kde");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");
  
  // query and reference datasets and target training values.
  Matrix references;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);

  KdeCV<GaussianKernel> kde_cv_algorithm;
  
  kde_cv_algorithm.Init(references);

  printf("Computed a Monte Carlo double sum of %g...\n",
	 kde_cv_algorithm.MonteCarloCompute());
  printf("Computed a naive double sum of %g...\n",
	 kde_cv_algorithm.NaiveCompute());

  // Finalize FastExec.
  fx_done(fx_root);
  return 0;
}
