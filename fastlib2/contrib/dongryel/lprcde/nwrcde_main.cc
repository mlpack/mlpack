/** @file nwrcde_main.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "nwrcde.h"

template<typename TKernel>
void StartComputation(const Matrix &queries, const Matrix &references,
		      const Matrix &reference_targets,
		      struct datanode *nwrcde_module) {

  // Output file name for the fast algorithm.
  const char *output_file_name = fx_param_str(nwrcde_module, "output",
                                              "nwrcde_results.txt");

  // Declare the computation object.
  NWRCde<TKernel> algorithm;
  NWRCdeQueryResult query_results;
  algorithm.Init(references, reference_targets, nwrcde_module);
  algorithm.Compute(queries, &query_results);
  query_results.PrintDebug(output_file_name);

  // If the do_naive flag is specified, then run the naive algorithm
  // as well.
  if(fx_param_exists(nwrcde_module, "do_naive")) {
    NWRCdeQueryResult naive_query_results;
    const char *naive_output_file_name =
      fx_param_str(nwrcde_module, "naive_output", "naive_nwrcde_results.txt");

    algorithm.NaiveCompute(queries, &naive_query_results);
    naive_query_results.PrintDebug(naive_output_file_name);

    // Compute the difference between the naively computed estimates
    // and the approximated estimates.
    Vector difference;
    la::SubInit(naive_query_results.final_nwr_estimates,
                query_results.final_nwr_estimates, &difference);

    // The maximum relative error.
    double max_relative_error = 0;
    int within_limit = 0;
    for(index_t i = 0; i < queries.n_cols(); i++) {
      double relative_error =
        (naive_query_results.final_nwr_estimates[i] ==
         query_results.final_nwr_estimates[i]) ?
        0:(fabs(difference[i]) /
           fabs(naive_query_results.final_nwr_estimates[i]));
      max_relative_error = std::max(max_relative_error, relative_error);

      if(relative_error < fx_param_double(nwrcde_module, "relative_error",
                                          0.1)) {
        within_limit++;
      }
    }
    fx_format_result(nwrcde_module, "max_relative_error", "%g",
                     max_relative_error);
    fx_format_result(nwrcde_module, "under_relative_error_limit", "%d",
                     within_limit);
  }
}

int main(int argc, char *argv[]) {
  
  // Initialize FastExec...
  fx_init(argc, argv, &nwrcde_main_doc);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "nwrcde_module" under the
  // root directory (NULL) for the NwrCde object to work inside.
  // Here, we initialize it with all parameters defined
  // "--nwrcde/...=...".
  struct datanode* nwrcde_module = fx_submodule(fx_root, "nwrcde");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");
  
  // The file containing the reference target values is a required
  // parameter.
  const char* reference_targets_file_name = fx_param_str_req
    (fx_root, "dtarget");

  // The query data file defaults to the references.
  const char* queries_file_name = fx_param_str(fx_root, "query",
					       references_file_name);

  // query and reference datasets and target training values.
  Matrix references;
  Matrix reference_targets;
  Matrix queries;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);  
  data::Load(queries_file_name, &queries);
  data::Load(reference_targets_file_name, &reference_targets);

  // Run the appropriate algorithm based on the kernel type.
  if(!strcmp(fx_param_str(nwrcde_module, "kernel", "gaussian"), "gaussian")) {
    StartComputation<GaussianKernel>(queries, references, reference_targets,
				     nwrcde_module);
  }
  else if(!strcmp(fx_param_str(nwrcde_module, "kernel", "epan"), "epan")) {
    StartComputation<EpanKernel>(queries, references, reference_targets,
				 nwrcde_module);
  }

  // Finalize FastExec and print output results.
  fx_done(fx_root);
  return 0;
}
