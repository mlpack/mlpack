#include "fastlib/fastlib.h"
#include "general_spacetree.h"
#include "contrib/dongryel/pca/pca.h"
#include "subspace_stat.h"
#include "gen_metric_tree.h"
#include "mlpack/kde/dataset_scaler.h"

typedef GeneralBinarySpaceTree<DBallBound < LMetric<2>, Vector>, Matrix, SubspaceStat > GTree;

int main(int argc, char *argv[]) {
 
  fx_init(argc, argv, NULL);
  const char *fname = fx_param_str(NULL, "data", NULL);
  Dataset dataset_;
  dataset_.InitFromFile(fname);
  Matrix data_;
  data_.Own(&(dataset_.matrix()));
  DatasetScaler::ScaleDataByMinMax(data_, data_, true);

  int leaflen = fx_param_int(NULL, "leaflen", 30);

  printf("Constructing the tree...\n");
  fx_timer_start(NULL, "pca tree");

  ArrayList<int> old_from_new;
  GTree *root_ = proximity::MakeGenMetricTree<GTree>
    (data_, leaflen, &old_from_new);
  
  fx_timer_stop(NULL, "pca tree");

  (root_->stat().singular_values_).PrintDebug();
  (root_->stat().left_singular_vectors_).PrintDebug();
  printf("Reconstruction error: %g\n",
	 (root_->stat()).max_l2_norm_reconstruction_error_);

  printf("Finished constructing the tree...\n");

  // exhaustively compute PCA
  printf("Exhaustive PCA\n");
  fx_timer_start(NULL, "exhaustive pca");
  Vector eigen_values;
  Matrix principal_components;
  Pca::EigenDecomposeCovariance(data_, &eigen_values, &principal_components);
  principal_components.PrintDebug();
  fx_timer_stop(NULL, "exhaustive pca");

  // Clean up the memory used by the tree...
  delete root_;

  fx_done(NULL);
  return 0;
}
