#include "fastlib/fastlib_int.h"
#include "dataset_scaler.h"
#include "kde.h"
#include "naive_kde.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////

  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(NULL, "data");
  
  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(NULL, "query", references_file_name);
  
  // flag for determining whether to compute naively
  bool do_naive = fx_param_exists(NULL, "do_naive");

  // FASTlib classes only poison data in their default constructors;
  // declarations must be followed by Init or an equivalent function.
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
  if(!strcmp(fx_param_str(NULL, "scaling", "none"), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references, 
				     queries_equal_references);
  }

  if(!strcmp(fx_param_str(NULL, "kernel", "gaussian"), "gaussian")) {
    
    Vector fast_kde_results;
    
    // for O(p^D) expansion
    if(fx_param_exists(NULL, "multiplicative_expansion")) {
      
      printf("O(p^D) expansion KDE\n");
      FastKde<GaussianKernelMultAux> fast_kde;
      fast_kde.Init(queries, references, queries_equal_references);
      fast_kde.Compute();
      
      if(fx_param_exists(NULL, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
      
      fast_kde.get_density_estimates(&fast_kde_results);
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      printf("O(D^p) expansion KDE\n");
      FastKde<GaussianKernelAux> fast_kde;
      fast_kde.Init(queries, references, queries_equal_references);
      fast_kde.Compute();
      
      if(fx_param_exists(NULL, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
      
      fast_kde.get_density_estimates(&fast_kde_results);
    }
    
    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(queries, references);
      naive_kde.Compute();
      
      if(fx_param_exists(NULL, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
    
  }
  else if(!strcmp(fx_param_str(NULL, "kernel", "epan"), "epan")) {
    FastKde<EpanKernelAux> fast_kde;
    fast_kde.Init(queries, references, queries_equal_references);
    fast_kde.Compute();
    
    if(fx_param_exists(NULL, "fast_kde_output")) {
      fast_kde.PrintDebug();
    }
    Vector fast_kde_results;
    fast_kde.get_density_estimates(&fast_kde_results);
    
    if(do_naive) {
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(queries, references);
      naive_kde.Compute();
      
      if(fx_param_exists(NULL, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
  }

  fx_done();
  return 0;
}
