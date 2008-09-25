#include "fast_multipole_method.h"

int main(int argc, char *argv[]) {
  
  fx_init(argc, argv, NULL);
  const char *fname = fx_param_str(NULL, "data", NULL);
  Dataset dataset_;
  dataset_.InitFromFile(fname);
  Matrix data_;
  data_.Own(&(dataset_.matrix()));

  // Declare the fast multipole method object.
  FastMultipoleMethod fmm_algorithm;
  
  fx_done(NULL);
  return 0;
}
