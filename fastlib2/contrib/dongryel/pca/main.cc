#include "fastlib/fastlib.h"
#include "pca.h"
#include "contrib/dongryel/proximity_project/subspace_stat.h"
#include "contrib/dongryel/proximity_project/gen_metric_tree.h"

typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, SubspaceStat > GTree;

void TestSvdLeftSingularVector(const Matrix &dataset) {

  Matrix principal_components;
  Vector eigen_values;
  
  fx_timer_start(fx_root, "svd");
  Pca::SvdLeftSingularVector(dataset, &eigen_values, &principal_components);
  fx_timer_stop(fx_root, "svd");
  
  principal_components.PrintDebug();
  eigen_values.PrintDebug();
}

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

  int leaflen = fx_param_int(NULL, "leaflen", 30);

  printf("Constructing the tree...\n");
  fx_timer_start(NULL, "svd_tree");

  ArrayList<int> old_from_new;
  GTree *root = proximity::MakeGenMetricTree<GTree>
    (dataset, leaflen, &old_from_new);

  fx_timer_stop(NULL, "svd_tree");
  printf("Finished constructing the tree...\n");

  delete root;

  fx_done();
  return 0;
}
