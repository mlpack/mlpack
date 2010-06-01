#include "fastlib/fastlib.h"
#include "compressed_matrix.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);

  CompressedVector<double, 100000000> cv;
  Vector v;
  index_t length_of_vector = 300000000;
  v.Init(length_of_vector);
  for(index_t i = 0; i < length_of_vector; i++) {
    v[i] = i;
  }

  // Copy the vector to the compressed one.
  cv.Copy(v);

  v.Destruct();
  // Measure the time for the uncompressed vector.
  //fx_timer_start(fx_root, "uncompressed_vector_dot_product");
  //dot_product = 0;
  //for(index_t i = 0; i < length_of_vector; i +=1000) {
  //dot_product += math::Sqr(v[i]);
  //}
  //fx_timer_stop(fx_root, "uncompressed_vector_dot_product");
  //printf("Dot product: %g\n", dot_product);

  // Measure the time for the compressed vector.
  fx_timer_start(fx_root, "compressed_vector_dot_product");
  double dot_product = 0;
  for(index_t i = 0; i < length_of_vector; i += 1000) {
    dot_product += math::Sqr(cv[i]);
  }
  fx_timer_stop(fx_root, "compressed_vector_dot_product");
  printf("Dot product: %g\n", dot_product);

  fx_done(fx_root);
  return 0;
}
