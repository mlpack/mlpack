#include "svm.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  Dataset dataset;
  
  //if (!PASSED(dataset.InitFromFile(fx_param_str_req(NULL, "data")))) {
  //  fprintf(stderr, "Couldn't open the data file.");
  //  return 1;
  //}
  Matrix m;
  index_t n = fx_param_int(NULL, "n", 30);
  double slope = fx_param_double(NULL, "slope", 1.0);
  double margin = fx_param_double(NULL, "margin", 1.0);
  double var = fx_param_double(NULL, "var", 1.0);
  
  m.Init(3, n);
  
  for (index_t i = 0; i < n; i += 2) {
    double x = (rand() * 2.0 / RAND_MAX) - 1.0;
    double y = margin / 2 + (rand() * var / RAND_MAX);
    m.set(0, i, x);
    m.set(1, i, x*slope+y);
    m.set(2, i, 0);
    m.set(0, i+1, x);
    m.set(1, i+1, x*slope-y);
    m.set(2, i+1, 1);
  }
  //Matrix m2;
  //la::TransposeInit(m, &m2);
  //m2.PrintDebug("m");
  
  dataset.AliasMatrix(m);
  
  SimpleCrossValidator< SVM<SVMLinearKernel> > cross_validator;
  cross_validator.Init(&dataset, 2, 4, fx_root, "svm");
  cross_validator.Run();
  
  fx_done();
}

