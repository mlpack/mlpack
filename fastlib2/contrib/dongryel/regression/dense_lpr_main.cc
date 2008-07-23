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

  printf("references dim: %d X %d \n", references.n_rows(), 
		references.n_cols());
  printf("reference_targets dim: %d X %d \n", reference_targets.n_rows(), 
		reference_targets.n_cols());
  printf("queries dim: %d X %d \n", queries.n_rows(), queries.n_cols());

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
    la::Scale(100.0, &reference_targets);	
    reference_targets.PrintDebug();
    
    printf("Running the DT-DENSE-LPR algorithm with relative prune rule.\n");
    DenseLpr<EpanKernel, RelativePruneLpr> fast_lpr;
    fast_lpr.Init(references, reference_targets, lpr_module);
    fast_lpr.PrintDebug();
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
	printf("before: queries dim: %d X %d \n", queries.n_rows(), queries.n_cols());

	fast_lpr.Compute(queries, &query_regression_estimates,
		 &query_confidence_bands, &query_magnitude_weight_diagrams);
        printf("after: queries dim: %d X %d \n", queries.n_rows(), queries.n_cols());

	index_t queries_length = queries.n_cols();
	printf("prediction on the queries (%d) done!\n", queries_length);

	const char* query_debugfile_name = 
		"estimates_1a0k_";
	FILE  *fp;
	fp = fopen(query_debugfile_name, "w");
	query_regression_estimates.PrintDebug("query_estimates", fp);
	for(index_t i = 0; i < query_regression_estimates.length(); i++) {
	  printf("%g\n", query_regression_estimates[i]);
	}
	double *ptr = query_regression_estimates.ptr();
	printf("%f\t%f\n", ptr[0], ptr[queries_length-1] );

	//Matrix query_regression_estimates_mat;
	//query_regression_estimates_mat.AliasColVector(*query_regression_estimates);
	const char* query_outfile_name = 
		"estimations_1a0k_";
	//data::Save(query_outfile_name, query_regression_estimates_mat);	
	fast_lpr.PrintDebug();

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

  // Do naive algorithm.
  /*
  printf("Running the naive algorithm...\n");
  Vector naive_lpr_results;
  NaiveLpr<EpanKernel> naive_lpr;
  ArrayList<DRange> naive_lpr_confidence_bands;
  naive_lpr.Init(references, reference_targets, lpr_module);
  naive_lpr.PrintDebug();
  naive_lpr.get_regression_estimates(&naive_lpr_results);
  naive_lpr.get_confidence_bands(&naive_lpr_confidence_bands);
  printf("Finished running the naive algorithm...\n");
  printf("Got root mean square deviation of %g...\n",
	 naive_lpr.root_mean_square_deviation());

  if(fast_lpr_has_run) {
    fx_format_result(lpr_module, "max_relative_difference_regression_estimate",
		     "%g", MatrixUtil::MaxRelativeDifference
		     (naive_lpr_results, fast_lpr_results));
    fx_format_result(lpr_module, "avg_relative_difference_regression_estimate",
		     "%g", MatrixUtil::AvgRelativeDifference
		     (naive_lpr_results, fast_lpr_results));

    fx_format_result
      (lpr_module, "relative_error_root_mean_square_deviation", "%g",
       fabs(fast_lpr_root_mean_square_deviation -
	    naive_lpr.root_mean_square_deviation()) /
       fabs(naive_lpr.root_mean_square_deviation()));

    DRange confidence_band_max_relative_difference =
      MatrixUtil::MaxRelativeDifference
      (naive_lpr_confidence_bands, fast_lpr_confidence_bands);
    DRange confidence_band_avg_relative_difference =
      MatrixUtil::AvgRelativeDifference
      (naive_lpr_confidence_bands, fast_lpr_confidence_bands);

    fx_format_result
      (lpr_module, "lower_quantile_avg_relative_error", "%g",
       confidence_band_avg_relative_difference.lo);
    fx_format_result
      (lpr_module, "lower_quantile_max_relative_error", "%g",
       confidence_band_max_relative_difference.lo);
    fx_format_result
      (lpr_module, "upper_quantile_avg_relative_error", "%g",
       confidence_band_avg_relative_difference.hi);
    fx_format_result
      (lpr_module, "upper_quantile_max_relative_error", "%g",
       confidence_band_max_relative_difference.hi);
  }
  else {

    // This is to prevent the code from complaining from uninitialized
    // object stuff.
    fast_lpr_results.Init(0);
    fast_lpr_confidence_bands.Init();
  }
  */

  // Finalize FastExec and print output results.
  fx_done(NULL);
  return 0;
}
