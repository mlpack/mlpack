#include "multibody_force_problem.h"
#include "multibody_kernel.h"
#include "../multitree_template/multitree_dfs.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec (parameter handling stuff).
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "fmm_module" under the
  // root directory (NULL) for the multibody object to work inside.
  // Here, we initialize it with all parameters defined
  // "--multibody/...=...".
  struct datanode *multibody_module = fx_submodule(fx_root, "multibody");
  
  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");

  // Query and reference datasets, reference weight dataset.
  Matrix references;
  
  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);


  la::Scale(references.n_cols() * references.n_rows(), 10.0, references.ptr());

  // Instantiate a multi-tree problem...
  ArrayList<const Matrix *> sets;
  sets.Init(AxilrodTellerForceProblem::order);
  for(index_t i = 0; i < AxilrodTellerForceProblem::order; i++) {
    sets[i] = &references;
  }

  MultiTreeDepthFirst<AxilrodTellerForceProblem> algorithm;
  AxilrodTellerForceProblem::MultiTreeQueryResult results;
  AxilrodTellerForceProblem::MultiTreeQueryResult naive_results;
  algorithm.InitMonoChromatic(sets, (const ArrayList<const Matrix *> *) NULL,
			      multibody_module);

  fx_timer_start(fx_root, "multitree");
  algorithm.Compute(NULL, &results);
  fx_timer_stop(fx_root, "multitree");

  results.PrintDebug("force_vectors.txt");
  printf("Got %d finite difference prunes...\n",
	 results.num_finite_difference_prunes);
  printf("Got %d Monte Carlo prunes...\n", results.num_monte_carlo_prunes);
  
  fx_timer_start(fx_root, "naive_code");
  algorithm.NaiveCompute((const ArrayList<const Matrix *> *) NULL,
			 &naive_results);
  fx_timer_stop(fx_root, "naive_code");
  naive_results.PrintDebug("naive_force_vectors.txt");
  double max_relative_error, positive_max_relative_error,
    negative_max_relative_error;
  naive_results.MaximumRelativeError(results, &max_relative_error,
				     &negative_max_relative_error,
				     &positive_max_relative_error);
  printf("Maximum relative error: %g\n", max_relative_error);
  printf("Positive max relative error: %g\n", positive_max_relative_error);
  printf("Negative max relative error: %g\n", negative_max_relative_error);

  fx_done(fx_root);
  return 0;
}
