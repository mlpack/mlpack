#include "mlpack/kde/dataset_scaler.h"
#include "dense_lpr.h"
#include "naive_lpr.h"
#include "relative_prune_lpr.h"

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
  Matrix *fast_numerator = NULL;
  ArrayList<Matrix> *fast_denominator = NULL;
  ArrayList<int> *old_from_new_queries = NULL;
  Vector fast_lpr_results;
  DenseLpr<GaussianKernel, 1, RelativePruneLpr> fast_lpr;
  fast_lpr.Init(queries, references, reference_targets, local_linear_module);
  fast_lpr.Compute();
  fast_lpr.PrintDebug();
  fast_lpr.get_regression_estimates(&fast_lpr_results);
  fast_lpr.get_intermediate_results(&fast_numerator, &fast_denominator,
				    &old_from_new_queries);

  // Do naive algorithm.
  ArrayList<Vector> *naive_numerator = NULL;
  ArrayList<Matrix> *naive_denominator = NULL;
  
  Vector naive_lpr_results;
  NaiveLpr<GaussianKernel, 1> naive_lpr;
  naive_lpr.Init(queries, references, reference_targets, local_linear_module);
  naive_lpr.Compute();
  naive_lpr.PrintDebug();
  naive_lpr.get_regression_estimates(&naive_lpr_results);
  naive_lpr.get_intermediate_results(&naive_numerator, &naive_denominator);
  printf("Maximum relative error: %g\n", 
	 MatrixUtil::MaxRelativeDifference(naive_lpr_results, 
					   fast_lpr_results));
  
  double max_relative_error = 0;
  for(index_t q = 0; q < queries.n_cols(); q++) {
    Vector q_fast_numerator;
    (*fast_numerator).MakeColumnVector(q, &q_fast_numerator);

    max_relative_error = 
      std::max(max_relative_error,
	       MatrixUtil::EntrywiseNormDifferenceRelative<Matrix>
	       ((*naive_denominator)[(*old_from_new_queries)[q]], 
		(*fast_denominator)[q], 1));
    max_relative_error = 
      std::max(max_relative_error,
	       MatrixUtil::EntrywiseNormDifferenceRelative<Vector>
	       ((*naive_numerator)[(*old_from_new_queries)[q]], 
		q_fast_numerator, 1));
  }
  printf("Matrix difference: %g\n", max_relative_error);

  // Finalize FastExec and print output results.
  fx_done();
  return 0;
}
