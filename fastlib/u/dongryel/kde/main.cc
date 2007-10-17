#include "fastlib/fastlib_int.h"
#include "fft_kde.h"
#include "kde.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  const char *algorithm = fx_param_str_req(NULL, "method");
  bool do_naive = fx_param_exists(NULL, "do_naive");
  const char *kernel_name = fx_param_str_req(NULL, "kernel");

  if(!strcmp(algorithm, "fft")) {
    FFTKde fft_kde;
    fft_kde.Init();
    fft_kde.Compute();
  }
  else {
    if(!strcmp(kernel_name, "gaussian")) {

      Vector fast_kde_results;
      Matrix query_dataset;
      Matrix reference_dataset;

      // for O(p^D) expansion
      if(fx_param_exists(NULL, "multiplicative_expansion")) {

	printf("O(p^D) expansion KDE\n");
	FastKde<GaussianKernel, GaussianKernelMultAux> fast_kde;
	fast_kde.Init();
	fast_kde.Compute(fx_param_double(NULL, "tau", 0.1));
      
	if(fx_param_exists(NULL, "fast_kde_output")) {
	  fast_kde.PrintDebug();
	}
      
	fast_kde_results.Copy(fast_kde.get_density_estimates());
	query_dataset.Copy(fast_kde.get_query_dataset());
	reference_dataset.Copy(fast_kde.get_reference_dataset());
      }

      // otherwise do O(D^p) expansion
      else {

	printf("O(D^p) expansion KDE\n");
	FastKde<GaussianKernel, GaussianKernelAux> fast_kde;
	fast_kde.Init();
	fast_kde.Compute(fx_param_double(NULL, "tau", 0.1));
      
	if(fx_param_exists(NULL, "fast_kde_output")) {
	  fast_kde.PrintDebug();
	}
      
	fast_kde_results.Copy(fast_kde.get_density_estimates());
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
    else if(!strcmp(kernel_name, "epan")) {
      FastKde<EpanKernel, EpanKernelAux> fast_kde;
      fast_kde.Init();
      fast_kde.Compute(fx_param_double(NULL, "tau", 0.1));
    
      if(fx_param_exists(NULL, "fast_kde_output")) {
	fast_kde.PrintDebug();
      }
    
      Vector fast_kde_results;
      fast_kde_results.Alias(fast_kde.get_density_estimates());
    
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
  }

  fx_done();
  return 0;
}
