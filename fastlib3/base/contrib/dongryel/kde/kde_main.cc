/** @file kde_main.cc
 *
 *  Kernel Density Estimator 2.0
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "contrib/dongryel/multitree_template/multitree_dfs.h"
#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "kde_problem.h"

template<typename TKernelAux>
void StartComputation(const Matrix &queries, const Matrix &references,
		      const Matrix &reference_targets,
		      struct datanode *module) {

  // Output file name for the fast algorithm.
  const char *output_file_name = fx_param_str(module, "output",
					      "kde_results.txt");

  // Declare the computation object.
  MultiTreeDepthFirst<KdeProblem<TKernelAux> > algorithm;
  typename KdeProblem<TKernelAux>::MultiTreeQueryResult query_results;

  // The array list of pointers to the query/reference sets and the
  // reference targets.
  ArrayList<const Matrix *> reference_sets;
  ArrayList<const Matrix *> query_sets;
  ArrayList<const Matrix *> reference_targets_set;
  
  // Nadaraya-Watson regression/conditional density estimation is a
  // two-body problem, hence there is only one reference set.
  reference_sets.Init(1);
  reference_sets[0] = &references;
  query_sets.Init(1);
  query_sets[0] = &queries;
  reference_targets_set.Init(1);
  reference_targets_set[0] = &reference_targets;
  
  algorithm.InitMultiChromatic(reference_sets, &reference_targets_set,
			       module);

  printf("Starting the multitree computation...\n");

  fx_timer_start(fx_root, "multitree_compute");
  algorithm.Compute(&query_sets, &query_results);
  fx_timer_stop(fx_root, "multitree_compute");
  printf("Finished the multitree computation...\n");

  query_results.PrintDebug(output_file_name);

  // If the do_naive flag is specified, then run the naive algorithm
  // as well.
  if(fx_param_exists(module, "do_naive")) {
    typename KdeProblem<TKernelAux>::MultiTreeQueryResult naive_query_results;
    const char *naive_output_file_name =
      fx_param_str(module, "naive_output", "naive_kde_results.txt");

    printf("Starting the naive computation...\n");
    fx_timer_start(fx_root, "naive_compute");
    algorithm.NaiveCompute(&query_sets, &naive_query_results);
    fx_timer_stop(fx_root, "naive_compute");
    printf("Finished the naive computation...\n");
    
    naive_query_results.PrintDebug(naive_output_file_name);

    // Compute the difference between the naively computed estimates
    // and the approximated estimates.
    Vector difference;
    la::SubInit(naive_query_results.final_results,
                query_results.final_results, &difference);

    // The maximum relative error.
    double max_relative_error = 0;
    int within_limit = 0;
    for(index_t i = 0; i < queries.n_cols(); i++) {
      double relative_error =
        (naive_query_results.final_results[i] ==
         query_results.final_results[i]) ?
        0:(fabs(difference[i]) /
           fabs(naive_query_results.final_results[i]));
      max_relative_error = std::max(max_relative_error, relative_error);

      if(relative_error <= fx_param_double(module, "relative_error", 0.1)) {
        within_limit++;
      }
    }
    fx_format_result(module, "max_relative_error", "%g", max_relative_error);
    fx_format_result(module, "under_relative_error_limit", "%d", within_limit);
  }

  // Output the prune statistics as well for the fast algorithm.
  fx_format_result(module, "num_finite_difference_prunes", "%d",
		   query_results.num_finite_difference_prunes);
  fx_format_result(module, "num_far_to_local_prunes", "%d",
		   query_results.num_far_to_local_prunes);
  fx_format_result(module, "num_direct_far_prunes", "%d",
		   query_results.num_direct_far_prunes);
  fx_format_result(module, "num_direct_local_prunes", "%d",
		   query_results.num_direct_local_prunes);
}

int main(int argc, char *argv[]) {
  
  // Initialize FastExec...
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode* kde_module = fx_submodule(fx_root, "kde");

  // Initialize the random number seed.
  srand(time(NULL));

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");

  // The query data file defaults to the references.
  const char* queries_file_name = fx_param_str(fx_root, "query",
					       references_file_name);

  bool queries_equal_references = !strcmp(references_file_name,
					  queries_file_name);

  // query and reference datasets and target training values.
  Matrix references;
  Matrix reference_targets;
  Matrix queries;

  // data::Load inits a matrix with the contents of a .csv or .arff.
  data::Load(references_file_name, &references);
  if(queries_equal_references) {
    queries.Alias(references);
  }
  else {
    data::Load(queries_file_name, &queries);
  }
  reference_targets.Init(references.n_cols(), 1);
  reference_targets.SetAll(1.0);

  // Confirm whether the user asked for scaling of the dataset
  if(!strcmp(fx_param_str(kde_module, "scaling", "none"), "range")) {
    printf("Range scaling...\n");
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }
  else if(!strcmp(fx_param_str(kde_module, "scaling", "none"), 
		  "standardize")) {
    printf("Standardized scaling...\n");
    DatasetScaler::StandardizeData(queries, references, 
				   queries_equal_references);
  }

  // Run the appropriate algorithm based on the kernel type.
  if(!strcmp(fx_param_str(kde_module, "kernel", "gaussian"), "gaussian")) {

    printf("Chose the Gaussian kernel...\n");
    StartComputation<GaussianKernelAux>(queries, references,
					reference_targets, kde_module);
  }
  else if(!strcmp(fx_param_str(kde_module, "kernel", "epan"), "epan")) {

    printf("Chose the Epanechnikov kernel...\n");
    StartComputation<EpanKernelAux>(queries, references, 
				    reference_targets, kde_module);
  }

  // Finalize FastExec and print output results.
  fx_done(fx_root);
  return 0;
}
