/** @file dualtree_kde_main.cc
 *
 *  Driver file for dualtree KDE algorithm.
 *
 *  @author Dongryeol Lee (dongryel)
 */

#include "fastlib/fastlib.h"
#include "dataset_scaler.h"
#include "dualtree_kde.h"
#include "dualtree_vkde.h"
#include "naive_kde.h"

/**
 * Main function which reads parameters and determines which
 * algorithms to run.
 *
 * In order to compile this driver, do:
 * fl-build dualtree_kde_bin --mode=fast
 *
 * In order to run this driver for the fast KDE algorithms, type the
 * following (which consists of both required and optional arguments)
 * in a single command line:
 *
 * ./dualtree_kde_bin --data=name_of_the_reference_dataset
 *                    --dwgts=name_of_the_reference_weight_dataset
 *                    --query=name_of_the_query_dataset 
 *                    --kde/kernel=gaussian
 *                    --kde/bandwidth=0.0130619
 *                    --kde/scaling=range 
 *                    --kde/multiplicative_expansion
 *                    --kde/fast_kde_output=fast_kde_output.txt
 *                    --kde/naive_kde_output=naive_kde_output.txt 
 *                    --kde/do_naive
 *                    --kde/relative_error=0.01
 *
 * Explanations for the arguments listed with possible values: 
 *
 * 1. data (required): the name of the reference dataset
 *
 * 2. dwgts (optional): the name of the reference weight dataset. If
 * missing, then assumes a uniformly weighted kernel density
 * estimation.
 *
 * 3. query (optional): the name of the query dataset (if missing, the
 * query dataset is assumed to be the same as the reference dataset)
 *
 * 4. kde/kernel (optional): kernel function to use
 * - gaussian: Gaussian kernel (default) 
 * - epan: Epanechnikov kernel
 *
 * 5. kde/bandwidth (required): smoothing parameter used for KDE; this
 * has to be positive.
 *
 * 6. kde/scaling (optional): whether to prescale the dataset 
 *
 * - range: scales both the query and the reference sets to be within
 * the unit hypercube [0, 1]^D where D is the dimensionality.
 * - standardize: scales both the query and the reference set to have
 * zero mean and unit variance.
 * - none: default value; no scaling
 *
 * 7. kde/multiplicative_expansion (optional): If this flag is
 * present, the series expansion for the Gaussian kernel uses O(p^D)
 * expansion. Otherwise, the Gaussian kernel uses O(D^p)
 * expansion. See kde.h for details.
 *
 * 8. kde/do_naive (optional): run the naive algorithm after the fast
 * algorithm.
 *
 * 9. kde/fast_kde_output (optional): if this flag is present, the
 * approximated density estimates are output to the filename provided
 * after it.
 *
 * 10. kde/naive_kde_output (optional): if this flag is present, the
 * exact density estimates computed by the naive algorithm are output
 * to the filename provided after it. This flag is not ignored if
 * --kde/do_naive flag is not present.
 *
 * 11. kde/relative_error (optional): relative error criterion for the
 * fast algorithm; default value is 0.1 (10 percent relative error for
 * all query density estimates).
 */
int main(int argc, char *argv[]) {

  // initialize FastExec (parameter handling stuff)
  fx_init(argc, argv, &kde_main_doc);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode* kde_module = fx_submodule(fx_root, "kde");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");

  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(fx_root, "query", references_file_name);
  
  // flag for determining whether to compute naively
  bool do_naive = fx_param_exists(kde_module, "do_naive");

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
    reference_weights.Init(1, queries.n_cols());
    reference_weights.SetAll(1);
  }
  
  // confirm whether the user asked for scaling of the dataset
  if(!strcmp(fx_param_str(kde_module, "scaling", "none"), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }
  else if(!strcmp(fx_param_str(kde_module, "scaling", "none"), 
		  "standardize")) {
    DatasetScaler::StandardizeData(queries, references, 
				   queries_equal_references);
  }

  if(!strcmp(fx_param_str(kde_module, "kernel", "gaussian"), "gaussian")) {
    
    Vector fast_kde_results;
    
    // for O(p^D) expansion
    if(fx_param_exists(kde_module, "multiplicative_expansion")) {
      
      printf("O(p^D) expansion KDE\n");
      DualtreeKde<GaussianKernelMultAux> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references, kde_module);
      fast_kde.Compute(&fast_kde_results);
      
      if(fx_param_exists(kde_module, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      printf("O(D^p) expansion KDE\n");
      DualtreeKde<GaussianKernelAux> fast_kde;
      fast_kde.Init(queries, references, reference_weights,
		    queries_equal_references, kde_module);
      fast_kde.Compute(&fast_kde_results);
      
      if(true || fx_param_exists(kde_module, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    }
    
    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights, kde_module);
      naive_kde.Compute();
      
      if(true || fx_param_exists(kde_module, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
    
  }
  else if(!strcmp(fx_param_str(kde_module, "kernel", "epan"), "epan")) {
    DualtreeKde<EpanKernelAux> fast_kde;
    Vector fast_kde_results;

    fast_kde.Init(queries, references, reference_weights,
		  queries_equal_references, kde_module);
    fast_kde.Compute(&fast_kde_results);
    
    if(fx_param_exists(kde_module, "fast_kde_output")) {
      fast_kde.PrintDebug();
    }
    
    if(do_naive) {
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(queries, references, reference_weights, kde_module);
      naive_kde.Compute();
      
      if(fx_param_exists(kde_module, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }

  fx_done(fx_root);
  return 0;
}
