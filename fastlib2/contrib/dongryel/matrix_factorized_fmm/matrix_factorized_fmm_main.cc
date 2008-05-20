#include "matrix_factorized_fmm.h"
#include "fastlib/fastlib.h"
#include "mlpack/kde/dataset_scaler.h"
#include "mlpack/series_expansion/matrix_factorized_kernel_aux.h"

int main(int argc, char *argv[]) {
  
  // Initialize FastExec (parameter handling stuff)
  fx_init(argc, argv);

  ////////// READING PARAMETERS AND LOADING DATA /////////////////////
  
  // FASTexec organizes parameters and results into submodules.  Think
  // of this as creating a new folder named "kde_module" under the
  // root directory (NULL) for the Kde object to work inside.  Here,
  // we initialize it with all parameters defined "--kde/...=...".
  struct datanode* kde_module =
    fx_submodule(NULL, "kde", "kde_module");
  
  // The reference data file is a required parameter.
  const char* references_file_name = fx_param_str_req(fx_root, "data");
  
  // The query data file defaults to the references.
  const char* queries_file_name =
    fx_param_str(fx_root, "query", references_file_name);
  
  // flag for determining whether to compute naively
  bool do_naive = fx_param_exists(kde_module, "do_naive");

  // The query and reference datasets
  Matrix references;
  Matrix queries;

  // The flag for telling whether references are equal to queries
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
  
  // Confirm whether the user asked for scaling of the dataset
  if(!strcmp(fx_param_str(kde_module, "scaling", "none"), "range")) {
    DatasetScaler::ScaleDataByMinMax(queries, references,
                                     queries_equal_references);
  }

  if(!strcmp(fx_param_str(kde_module, "kernel", "gaussian"), "gaussian")) {
    
    Vector fast_kde_results;
    
    printf("Kernel independent expansion for Gaussian kernel KDE\n");
    MatrixFactorizedFMM<GaussianKernelMatrixFactorizedAux> fast_kde;
    fast_kde.Init(references, kde_module);
    fast_kde.Compute(queries, &fast_kde_results);
  }
  else if(!strcmp(fx_param_str(kde_module, "kernel", "epan"), "epan")) {
    MatrixFactorizedFMM<EpanKernelMatrixFactorizedAux> fast_kde;
    Vector fast_kde_results;

    fast_kde.Init(references, kde_module);
    fast_kde.Compute(queries, &fast_kde_results);
  }

  fx_done();
  return 0;
}
