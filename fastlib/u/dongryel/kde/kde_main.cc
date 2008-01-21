#include "fastlib/fastlib_int.h"
#include "dataset_scaler.h"
#include "kde.h"
#include "naive_kde.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode* kde_module =
    fx_submodule(NULL, "kde", "kde_module");

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "data");
  
  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(NULL, "query", references_file_name);
  
  // flag for determining whether to compute naively
  bool do_naive = fx_param_exists(kde_module, "do_naive");

  // query and reference datasets
  Matrix references;
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
  
  // confirm whether the user asked for scaling of the dataset
  if(!strcmp(fx_param_str(kde_module, "scaling", "none"), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }

  if(!strcmp(fx_param_str(kde_module, "kernel", "gaussian"), "gaussian")) {
    
    Vector fast_kde_results;
    
    // for O(p^D) expansion
    if(fx_param_exists(kde_module, "multiplicative_expansion")) {
      
      printf("O(p^D) expansion KDE\n");
      FastKde<GaussianKernelMultAux> fast_kde;
      fast_kde.Init(queries, references, queries_equal_references, 
		    kde_module);
      fast_kde.Compute();
      
      if(fx_param_exists(kde_module, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
      
      fast_kde.get_density_estimates(&fast_kde_results);
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      printf("O(D^p) expansion KDE\n");
      FastKde<GaussianKernelAux> fast_kde;
      fast_kde.Init(queries, references, queries_equal_references,
		    kde_module);
      fast_kde.Compute();
      
      if(fx_param_exists(kde_module, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
      
      fast_kde.get_density_estimates(&fast_kde_results);
    }
    
    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references, kde_module);
      naive_kde.Compute();
      
      if(fx_param_exists(kde_module, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
    
  }
  else if(!strcmp(fx_param_str(kde_module, "kernel", "epan"), "epan")) {
    FastKde<EpanKernelAux> fast_kde;
    fast_kde.Init(queries, references, queries_equal_references, kde_module);
    fast_kde.Compute();
    
    if(fx_param_exists(kde_module, "fast_kde_output")) {
      fast_kde.PrintDebug();
    }
    Vector fast_kde_results;
    fast_kde.get_density_estimates(&fast_kde_results);
    
    if(do_naive) {
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(queries, references, kde_module);
      naive_kde.Compute();
      
      if(fx_param_exists(kde_module, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }

  fx_done();
  return 0;
}
