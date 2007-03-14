/**
 * @file main.cc
 *
 * Test driver for our k-nearest-neighbors classifier example.
 */

#include "helper.h"

#include "fastlib/fastlib.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);

  const char *data = fx_param_str(NULL, "data", NULL);
  int n_labels = fx_param_int(NULL, "n_labels", 0);

  Dataset dataset;
  
  if (!PASSED(dataset.InitFromFile(data))) {
    fprintf(stderr, "main: Couldn't open file '%s'.\n", data);
    return 1;
  }
  
  SimpleCrossValidator<KnnClassifier> cross_validator;
  
  cross_validator.Init(&dataset, n_labels, 10, fx_root, "knn");
  
  cross_validator.Run();
  
  fx_done();
}
