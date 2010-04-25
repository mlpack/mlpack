#include "fastlib/fastlib.h"
#include "krylov_lpr.h"
#include "mlpack/kde/dataset_scaler.h"
#include "naive_lpr.h"
#include "relative_prune_lpr.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec...
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined
  // "--lpr/...=...".
  struct datanode* krylov_lpr_module =
    fx_submodule(NULL, "lpr");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str
    (NULL, "data", "alldata_deltacolors_stdized");
  
  // The file containing the reference target values is a required
  // parameter.
  const char* reference_targets_file_name = 
    fx_param_str(NULL, "dtarget", "alldata_zs");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(NULL, "query", references_file_name);

  // query and reference datasets and target training values.
  Matrix references;
  Matrix reference_targets;
  Matrix queries;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);  
  data::Load(queries_file_name, &queries);
  data::Load(reference_targets_file_name, &reference_targets);

  // We assume that the reference dataset lies in the positive
  // quadrant for simplifying the algorithmic implementation. Scale
  // the datasets to fit in the hypercube. This should be replaced
  // with more general dataset scaling operation, requested by the
  // users.
  DatasetScaler::ScaleDataByMinMax(queries, references, false);

  // Declare local linear krylov object.
  Vector fast_lpr_results;
  KrylovLpr<EpanKernel, RelativePruneLpr> fast_lpr;
  fast_lpr.Init(references, reference_targets, krylov_lpr_module);
  fast_lpr.get_regression_estimates(&fast_lpr_results);
  fast_lpr.PrintDebug();

  // Do naive algorithm.
  Vector naive_lpr_results;
  NaiveLpr<EpanKernel> naive_lpr;
  naive_lpr.Init(references, reference_targets, krylov_lpr_module);
  naive_lpr.PrintDebug();
  naive_lpr.get_regression_estimates(&naive_lpr_results);
  printf("Finished running the naive algorithm...\n");

  printf("Maximum relative difference in regression estimate: %g\n",
	 MatrixUtil::MaxRelativeDifference(naive_lpr_results,
					   fast_lpr_results));

  // Finalize FastExec and print output results.
  fx_done(NULL);
  return 0;
}
