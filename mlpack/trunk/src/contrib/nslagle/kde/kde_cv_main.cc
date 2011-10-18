#include "kde_cv.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec.
  CLI::ParseCommandLine (argc, argv);

  // Initialize the random number seed.
  srand(time(NULL));

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  //struct datanode* kde_module = fx_submodule(fx_root, "kde");

  // The reference data file is a required parameter.
  arma::mat references;
  std::string references_filename = CLI::GetParam<std::string>("kde/data");

  // query and reference datasets and target training values.
  // data::Load inits a matrix with the contents of a .csv or .arff.
  if (data::Load (references_filename.c_str(), references) == false)
  {
    Log::Fatal << "Reference file " << references_filename << " not found." << std::endl;
  }

  Log::Info << "Loaded reference data from " << references_filename << std::endl;

  KdeCV<GaussianKernel> kde_cv_algorithm;

  kde_cv_algorithm.Init(references);

  printf("Computed a Monte Carlo double sum of %g...\n",
	 kde_cv_algorithm.MonteCarloCompute());
  printf("Computed a naive double sum of %g...\n",
	 kde_cv_algorithm.NaiveCompute());

  // Finalize FastExec.
  //fx_done(fx_root);
  return 0;
}
