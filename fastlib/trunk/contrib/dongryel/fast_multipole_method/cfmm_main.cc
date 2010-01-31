#include "continuous_fmm.h"

int main(int argc, char *argv[]) {

  // Initialize FastExec (parameter handling stuff).
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "cfmm_module" under the
  // root directory (NULL) for the continuous FMM object to work
  // inside.  Here, we initialize it with all parameters defined
  // "--cfmm/...=...".
  struct datanode* cfmm_module = fx_submodule(fx_root, "cfmm");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(fx_root, "query", references_file_name);
  
  // Query and reference datasets, reference weight dataset, and the
  // reference bandwidth dataset.
  Matrix references;
  Matrix reference_weights;
  Matrix reference_bandwidths;
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

  // Read in the bandwidth for each point.
  if(fx_param_exists(fx_root, "bandwidths")) {
    data::Load(fx_param_str(fx_root, "bandwidths", NULL),
	       &reference_bandwidths);
  }
  else {
    reference_bandwidths.Init(1, references.n_cols());
    reference_bandwidths.SetAll(1);
  }

  // Declare the fast multipole method object and initialize it.
  ContinuousFmm cfmm_algorithm;
  cfmm_algorithm.Init(queries, references, reference_weights,
		      reference_bandwidths, queries_equal_references, 
		      cfmm_module);
  
  // Start computation.
  Vector cfmm_results;
  cfmm_algorithm.Compute(&cfmm_results);

  // Compute naively.
  Vector naive_results;
  cfmm_algorithm.NaiveCompute(&naive_results);

  fx_done(NULL);
  return 0;
}
