#include "fastlib/fastlib_int.h"
#include "kde.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  bool do_naive = fx_param_exists(NULL, "do_naive");

  FastKde<GaussianKernel, GaussianKernelDerivative> fast_kde;
  fast_kde.Init();
  fast_kde.Compute(fx_param_double(NULL, "tau", 0.1));
  //fast_kde.PrintDebug();
  Vector fast_kde_results;
  fast_kde_results.Alias(fast_kde.get_density_estimates());
  
  if(do_naive) {
    NaiveKde<GaussianKernel> naive_kde;
    naive_kde.Init(fast_kde.get_query_dataset(), 
		   fast_kde.get_reference_dataset());
    naive_kde.Compute();
    //naive_kde.PrintDebug();
    naive_kde.ComputeMaximumRelativeError(fast_kde_results);
  }

  fx_done();
  return 0;
}
