#include "fastlib/fastlib_int.h"
#include "pca_tree.h"
#include "spill_kdtree.h"

typedef GeneralBinarySpaceTree<DHrectBound<2>, Matrix, PCAStat> Tree;

void PCA(const Matrix &data) {
  
  // compute PCA on the extracted submatrix
  Matrix U, VT;
  Vector s;
  la::SVDInit(data, &s, &U, &VT);

  // reduce the dimension in half
  Matrix U_trunc;
  int new_dimension = U.n_cols();
  U_trunc.Init(new_dimension, U.n_rows());
  for(index_t i = 0; i < new_dimension; i++) {
    Vector s;
    U.MakeColumnVector(i, &s);

    for(index_t j = 0; j < U.n_rows(); j++) {
      U_trunc.set(i, j, s[j]);
    }
  }

  Matrix pca_transformed;
  la::MulInit(U_trunc, data, &pca_transformed);

  pca_transformed.PrintDebug();
}

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  const char *fname = fx_param_str(NULL, "data", NULL);
  Dataset dataset_;
  dataset_.InitFromFile(fname);
  Matrix data_;
  data_.Own(&(dataset_.matrix()));
  int leaflen = fx_param_int(NULL, "leaflen", 20);
  Tree *root_ = 
    proximity::MakeSpillKdTreeMidpoint<Tree>(data_, leaflen, NULL);

  // recursively computed PCA
  printf("Recursive PCA\n");
  (root_->stat().pca_transformed_).PrintDebug();

  // exhaustively compute PCA
  printf("Exhaustive PCA\n");
  PCA(data_);

  fx_done();
  return 0;
}
