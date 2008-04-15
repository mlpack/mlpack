#include "fastlib/fastlib.h"
#include "pca.h"

void TestFixedPointAlgorithm(const Matrix &dataset) {
  Matrix principal_components;
  Vector eigen_values;
  
  fx_timer_start(fx_root, "fixed_point_algorithm");
  Pca::FixedPointAlgorithm(dataset, &eigen_values, &principal_components,
			   dataset.n_rows() / 2, 0.01);
  fx_timer_stop(fx_root, "fixed_point_algorithm");

  principal_components.PrintDebug();

  eigen_values.PrintDebug();
}

void TestEigenDecomposeCovariance(const Matrix &dataset) {
  Matrix principal_components;
  Vector eigen_values;

  fx_timer_start(fx_root, "eigen_decompose_covariance");
  Pca::EigenDecomposeCovariance(dataset, &eigen_values, &principal_components);
  fx_timer_stop(fx_root, "eigen_decompose_covariance");

  principal_components.PrintDebug();

  eigen_values.PrintDebug();
}

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv);
  const char *file_name = fx_param_str(NULL, "data", NULL);
  Matrix dataset;
  data::Load(file_name, &dataset);
  
  TestEigenDecomposeCovariance(dataset);

  TestFixedPointAlgorithm(dataset);

  fx_done();
  return 0;
}
