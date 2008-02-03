#include "fastlib/fastlib.h"
#include "local_linear_krylov.h"

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
  const char* references_file_name = fx_param_str_req(NULL, "data");
  
  // The file containing the reference target values is a required
  // parameter.
  const char* reference_targets_file_name = fx_param_str_req(NULL, "dtarget");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(NULL, "query", references_file_name);

  // query and reference datasets and target training values.
  Matrix references;
  Matrix reference_targets;
  Matrix queries;

  // flag for telling whether references are equal to queries
  bool queries_equal_references = 
    !strcmp(queries_file_name, references_file_name);

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);  
  if(queries_equal_references) {
    queries.Alias(references);
  }
  else {
    data::Load(queries_file_name, &queries);
  }
  data::Load(reference_targets_file_name, &reference_targets);

  // Declare local linear krylov object.
  LocalLinearKrylov<GaussianKernel> local_linear;
  local_linear.Init(queries, references, reference_targets,
		    queries_equal_references, local_linear_module);
  local_linear.Compute();

  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}
