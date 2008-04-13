#include "fastlib/fastlib.h"
#include "lanczos_pca.h"

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv);
  const char *file_name = fx_param_str(NULL, "data", NULL);
  Matrix dataset;
  data::Load(file_name, &dataset);

  // declare Lanczos-based PCA
  Matrix principal_components;
  LanczosPca::Compute(dataset, &principal_components);
  
  fx_done();
  return 0;
}
