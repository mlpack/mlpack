#include "fastlib/fastlib_int.h"
#include "fft_kde.h"
#include "kde.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  FFTKde fft_kde;
  Vector fft_kde_results;
  Matrix query_dataset;
  Matrix reference_dataset;
  fft_kde.Init();
  fft_kde.Compute();
  
  fft_kde_results.Copy(fft_kde.get_density_estimates());
  query_dataset.Copy(fft_kde.get_query_dataset());
  reference_dataset.Copy(fft_kde.get_reference_dataset());
  if(fx_param_exists(NULL, "fft_kde_output")) {
    fft_kde.PrintDebug();
  }
  
  if(fx_param_exists(NULL, "do_naive")) {
    NaiveKde<GaussianKernel> naive_kde;
    naive_kde.Init(query_dataset, reference_dataset);
    naive_kde.Compute();
    
    if(fx_param_exists(NULL, "naive_kde_output")) {
      naive_kde.PrintDebug();
    }
    naive_kde.ComputeMaximumRelativeError(fft_kde_results);
  }

  fx_done();
}
