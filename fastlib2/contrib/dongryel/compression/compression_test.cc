#include "fastlib/fastlib.h"
#include "compressed_matrix.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);

  CompressedVector<double> cv;
  Vector v;
  index_t length_of_vector = 3000;
  v.Init(length_of_vector);
  for(index_t i = 0; i < length_of_vector; i++) {
    v[i] = i;
  }
  cv.Copy(v);

  for(index_t i = 0; i < length_of_vector; i++) {
    printf("I got %g\n", cv[i]);
  }
  fx_done(fx_root);
  return 0;
}
