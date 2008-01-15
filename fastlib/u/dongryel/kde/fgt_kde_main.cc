#include "fastlib/fastlib_int.h"
#include "fgt_kde.h"
#include "kde.h"
#include "naive_kde.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  FGTKde fgt_kde;
  Vector fgt_kde_results;
  Matrix query_dataset;
  Matrix reference_dataset;
  fgt_kde.Init();    
  fgt_kde.Compute();
  fgt_kde_results.Copy(fgt_kde.get_density_estimates());
  query_dataset.Copy(fgt_kde.get_query_dataset());
  reference_dataset.Copy(fgt_kde.get_reference_dataset());
  if(fx_param_exists(NULL, "fgt_kde_output")) {
    fgt_kde.PrintDebug();
  }
  
  if(fx_param_exists(NULL, "do_naive")) {
    NaiveKde<GaussianKernel> naive_kde;
    naive_kde.Init(query_dataset, reference_dataset);
    naive_kde.Compute();
    
    if(fx_param_exists(NULL, "naive_kde_output")) {
      naive_kde.PrintDebug();
    }
    naive_kde.ComputeMaximumRelativeError(fgt_kde_results);
  }

  fx_done();
}
