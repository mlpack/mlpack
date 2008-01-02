#include "fastlib/fastlib.h"

void whiten(Matrix X, Matrix &X_whitened) {
  Matrix X_transpose, X_squared, D, E, E_transpose, X_temp, X_temp2;
  Vector D_vector;

  la::TransposeInit(X, &X_transpose);
  la::MulInit(X, X_transpose, &X_squared);

  la::EigenvectorsInit(X_squared, &D_vector, &E);
  la::TransposeInit(E, &E_transpose);
  D.InitDiagonal(D_vector);
  int n_dims = D.n_rows();
  for(int i = 0; i < n_dims; i++) {
    D.set(i, i, pow(D.get(i, i), -.5));
  }

  la::MulInit(E, D, &X_temp);
  la::MulInit(X_temp, E_transpose, &X_temp2);
  la::MulInit(X_temp2, X, &X_whitened);
}


int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *data = fx_param_str(NULL, "data", NULL);

  Matrix X, whitenedX;
  data::Load(data, &X);

  index_t n_dims = X.n_rows();
  index_t n_points = X.n_cols();

  whiten(X, &X_whitened);


  fx_done();

  return 0;
}
