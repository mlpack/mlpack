#include "fastlib/fastlib.h"
#include "pca.h"

void TestEigenDecomposeCovariance(const Matrix &dataset) {
  Matrix principal_components;
  Vector eigen_values;
  Pca::EigenDecomposeCovariance(dataset, &eigen_values, &principal_components);

  principal_components.PrintDebug();

  eigen_values.PrintDebug();
}

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv);
  const char *file_name = fx_param_str(NULL, "data", NULL);
  Matrix dataset;
  data::Load(file_name, &dataset);
  
  TestEigenDecomposeCovariance(dataset);

  fx_done();
  return 0;
}
