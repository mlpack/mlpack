#include "fastlib/fastlib.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *data = fx_param_str(NULL, "data", NULL);

  Matrix X;
  data::Load(data, &X);

  index_t n_dims = X.n_rows();
  index_t n_points = X.n_cols();

  fx_done();

  return 0;
}
