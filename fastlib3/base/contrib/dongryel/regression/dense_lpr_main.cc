/** @file dense_lpr_main.cc
 *
 *  @author Dongryeol Lee (dongryel@cc.gatech.edu)
 */

#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/kernel_aux.h"
#include "dense_lpr.h"
#include "naive_lpr.h"
#include "quick_prune_lpr.h"
#include "relative_prune_lpr.h"

int main(int argc, char *argv[]) {
  
  // Initialize FastExec...
  fx_init(argc, argv, NULL);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Lpr object to work inside.  Here,
  // we initialize it with all parameters defined "--lpr/...=...".
  struct datanode* lpr_module =
    fx_submodule(NULL, "lpr");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str
    (NULL, "data", 
     "references1.txt");
  
  // The file containing the reference target values is a required
  // parameter.
  const char* reference_targets_file_name = 
    fx_param_str
    (NULL, "dtarget", 
     "targets1.txt");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(NULL, "query", 
    "queries_1a0k_.txt");

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
  //DatasetScaler::ScaleDataByMinMax(queries, references, false);

  // Store the results computed by the tree-based results.
  Vector fast_lpr_results;
  ArrayList<DRange> fast_lpr_confidence_bands;
  double fast_lpr_root_mean_square_deviation = 0;
  bool fast_lpr_has_run = false;
  if(!strcmp(fx_param_str_req(lpr_module, "method"), "dt-dense-quick")) {
    printf("Running the DT-DENSE-LPR with Deng and Moore's prune rule.\n");
    DenseLpr<EpanKernel, QuickPruneLpr> fast_lpr;
    fast_lpr.Init(references, reference_targets, lpr_module);
    fast_lpr.PrintDebug();
    fast_lpr.get_regression_estimates(&fast_lpr_results);
    printf("Finished the DT-DENSE-LPR with Deng and Moore's prune rule.\n");
    printf("Root mean square deviation of %g...\n",
	   (fast_lpr_root_mean_square_deviation = 
	    fast_lpr.root_mean_square_deviation()));
    fast_lpr_has_run = true;
    fast_lpr.get_confidence_bands(&fast_lpr_confidence_bands);
  }
  else if(!strcmp(fx_param_str_req(lpr_module, "method"), "st-dense-quick")) {
    printf("Running the ST-DENSE-LPR with Deng and Moore's prune rule.\n");
    DenseLpr<EpanKernel, QuickPruneLpr> fast_lpr;
    fast_lpr.Init(references, reference_targets, lpr_module);
    fast_lpr.PrintDebug();
    fast_lpr.get_regression_estimates(&fast_lpr_results);
    printf("Finished the ST-DENSE-LPR with Deng and Moore's prune rule.\n");
    printf("Root mean square deviation of %g...\n",
	   (fast_lpr_root_mean_square_deviation = 
	    fast_lpr.root_mean_square_deviation()));
    fast_lpr_has_run = true;
    fast_lpr.get_confidence_bands(&fast_lpr_confidence_bands);
  }
  else if(!strcmp(fx_param_str_req(lpr_module, "method"), 
		  "dt-dense-relative")) {
    printf("Running the DT-DENSE-LPR algorithm with relative prune rule.\n");
    DenseLpr<EpanKernel, RelativePruneLpr> fast_lpr;
    fast_lpr.Init(references, reference_targets, lpr_module);
    fast_lpr.get_regression_estimates(&fast_lpr_results);
    printf("Finished the DT-DENSE-LPR algorithm with relative prune rule.\n");
    printf("Root mean square deviation of %g...\n",
	   (fast_lpr_root_mean_square_deviation = 
	    fast_lpr.root_mean_square_deviation()));
    fast_lpr_has_run = true;
    fast_lpr.get_confidence_bands(&fast_lpr_confidence_bands);
    printf("Model Training Done!\n");
    
    Vector query_regression_estimates ;
    ArrayList<DRange> query_confidence_bands;
    Vector query_magnitude_weight_diagrams;
    fast_lpr.Compute(queries, &query_regression_estimates,
		     &query_confidence_bands, &query_magnitude_weight_diagrams);
    
    index_t queries_length = queries.n_cols();
    
    const char* query_debugfile_name = 
      fx_param_str(fx_root, "query_estimate_output_file", "query_estimates");
    FILE  *fp;
    fp = fopen(query_debugfile_name, "w");
    for(index_t i = 0; i < query_regression_estimates.length(); i++) {
      fprintf(fp, "%g\n", query_regression_estimates[i]);
    }
    fclose(fp);
  }
  else if(!strcmp(fx_param_str_req(lpr_module, "method"), 
		  "st-dense-relative")) {

    printf("Running the ST-DENSE-LPR algorithm with relative prune rule.\n");
    DenseLpr<EpanKernel, RelativePruneLpr> fast_lpr;
    fast_lpr.Init(references, reference_targets, lpr_module);
    fast_lpr.PrintDebug();
    fast_lpr.get_regression_estimates(&fast_lpr_results);
    printf("Finished the ST-DENSE-LPR algorithm with relative prune rule.\n");
    printf("Root mean square deviation of %g...\n",
	   (fast_lpr_root_mean_square_deviation = 
	    fast_lpr.root_mean_square_deviation()));
    fast_lpr_has_run = true;
    fast_lpr.get_confidence_bands(&fast_lpr_confidence_bands);
  }


  // Finalize FastExec and print output results.
  fx_done(NULL);
  return 0;
}
