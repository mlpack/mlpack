#include "mlpack/kde/dataset_scaler.h"
#include "dense_lpr.h"

int main(int argc, char *argv[]) {
  
  // Initialize FastExec...
  fx_init(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined
  // "--local_linear/...=...".
  struct datanode* local_linear_module =
    fx_submodule(NULL, "local_linear", "local_linear_module");

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
  DatasetScaler::TranslateDataByMin(queries, references, false);

  // Declare local linear krylov object.
  DenseLpr<GaussianKernel> local_linear;
  local_linear.Init(queries, references, reference_targets, 
		    local_linear_module);
  local_linear.Compute();
  local_linear.PrintDebug();
  
  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}
