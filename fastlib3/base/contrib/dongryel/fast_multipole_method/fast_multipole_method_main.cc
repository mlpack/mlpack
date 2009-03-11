#include "fast_multipole_method.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec (parameter handling stuff).
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "fmm_module" under the
  // root directory (NULL) for the FMM object to work inside.  Here,
  // we initialize it with all parameters defined "--fmm/...=...".
  struct datanode* fmm_module = fx_submodule(fx_root, "fmm");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(fx_root, "query", references_file_name);
  
  // Query and reference datasets, reference weight dataset.
  Matrix references;
  Matrix reference_weights;
  Matrix queries;

  // Flag for telling whether references are equal to queries
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

  // If the reference weight file name is specified, then read in,
  // otherwise, initialize to uniform weights.
  if(fx_param_exists(fx_root, "dwgts")) {
    data::Load(fx_param_str(fx_root, "dwgts", NULL), &reference_weights);
  }
  else {
    reference_weights.Init(1, references.n_cols());
    reference_weights.SetAll(1);
  }

  // Declare the fast multipole method object and initialize it.
  FastMultipoleMethod fmm_algorithm;
  fmm_algorithm.Init(queries, references, reference_weights,
		     queries_equal_references, fmm_module);
  
  // Start computation.
  fmm_algorithm.Compute();

  Vector naive_results;
  fmm_algorithm.NaiveCompute(&naive_results);

  fx_done(NULL);
  return 0;
}
