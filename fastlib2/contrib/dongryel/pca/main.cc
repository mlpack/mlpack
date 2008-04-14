#include "fastlib/fastlib.h"
#include "pca.h"

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv);
  const char *file_name = fx_param_str(NULL, "data", NULL);
  Matrix dataset;
  data::Load(file_name, &dataset);

  Matrix principal_components;
  Vector eigen_values;
  Pca::EigenDecomposeCovariance(dataset, &eigen_values, &principal_components);
  
  fx_done();
  return 0;
}
