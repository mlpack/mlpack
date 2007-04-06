#include "svm.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  Dataset dataset;
  
  if (fx_param_exists(NULL, "data")) {
    // if a data file is specified, use it.
    if (!PASSED(dataset.InitFromFile(fx_param_str_req(NULL, "data")))) {
      fprintf(stderr, "Couldn't open the data file.");
      return 1;
    }
  } else {
    Matrix m;
    index_t n = fx_param_int(NULL, "n", 30);
    double slope = fx_param_double(NULL, "slope", 1.0);
    double margin = fx_param_double(NULL, "margin", 1.0);
    double var = fx_param_double(NULL, "var", 1.0);
    double intercept = fx_param_double(NULL, "intercept", 1.0);
    
    m.Init(3, n);
    
    for (index_t i = 0; i < n; i += 2) {
      double x;
      double y;
      
      x = (rand() * 1.0 / RAND_MAX) + 1.0;
      y = margin / 2 + (rand() * var / RAND_MAX);
      m.set(0, i, x);
      m.set(1, i, x*slope + y + intercept);
      m.set(2, i, 0);
      
      x = (rand() * 1.0 / RAND_MAX) + 1.0;
      y = margin / 2 + (rand() * var / RAND_MAX);
      m.set(0, i+1, x);
      m.set(1, i+1, x*slope - y + intercept);
      m.set(2, i+1, 1);
    }
    dataset.OwnMatrix(&m);
  }
  
  SimpleCrossValidator< SVM<SVMLinearKernel> > cross_validator;
  cross_validator.Init(&dataset, 2, 4, fx_root, "svm");
  cross_validator.Run(true);
  
  fx_done();
}

