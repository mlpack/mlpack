#include "fastlib/fastlib_int.h"
#include "kde.h"
#include "naive_kde.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  bool do_naive = fx_param_exists(NULL, "do_naive");
  
  if(!strcmp(fx_param_str(NULL, "kernel", "gaussian"), "gaussian")) {
    
    Vector fast_kde_results;
    Matrix query_dataset;
    Matrix reference_dataset;
    
    // for O(p^D) expansion
    if(fx_param_exists(NULL, "multiplicative_expansion")) {
      
      printf("O(p^D) expansion KDE\n");
      FastKde<GaussianKernelMultAux> fast_kde;
      fast_kde.Init();
      fast_kde.Compute();
      
      if(fx_param_exists(NULL, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
      
      fast_kde.get_density_estimates(&fast_kde_results);
      query_dataset.Copy(fast_kde.get_query_dataset());
      reference_dataset.Copy(fast_kde.get_reference_dataset());
    }
    
    // otherwise do O(D^p) expansion
    else {
      
      printf("O(D^p) expansion KDE\n");
      FastKde<GaussianKernelAux> fast_kde;
      fast_kde.Init();
      fast_kde.Compute();
      
      if(fx_param_exists(NULL, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
      
      fast_kde.get_density_estimates(&fast_kde_results);
      query_dataset.Copy(fast_kde.get_query_dataset());
      reference_dataset.Copy(fast_kde.get_reference_dataset());
    }
    
    if(do_naive) {
      NaiveKde<GaussianKernel> naive_kde;
      naive_kde.Init(query_dataset, reference_dataset);
      naive_kde.Compute();
      
      if(fx_param_exists(NULL, "naive_kde_output")) {
	naive_kde.PrintDebug();
      }
      naive_kde.ComputeMaximumRelativeError(fast_kde_results);
    }
    
  }
  else if(!strcmp(fx_param_str(NULL, "kernel", "epan"), "epan")) {
    FastKde<EpanKernelAux> fast_kde;
    fast_kde.Init();
    fast_kde.Compute();
    
    if(fx_param_exists(NULL, "fast_kde_output")) {
      fast_kde.PrintDebug();
    }
    Vector fast_kde_results;
    fast_kde.get_density_estimates(&fast_kde_results);
    
    if(do_naive) {
      NaiveKde<EpanKernel> naive_kde;
      naive_kde.Init(fast_kde.get_query_dataset(),
		     fast_kde.get_reference_dataset());
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
