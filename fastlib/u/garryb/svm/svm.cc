#include "svm.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  
  Dataset dataset;
  
  if (!PASSED(dataset.InitFromFile(fx_param_str_req(NULL, "data")))) {
    fprintf(stderr, "Couldn't open the data file.");
    return 1;
  }
  
  SimpleCrossValidator< SVM<SVMLinearKernel> > cross_validator;
  cross_validator.Init(&dataset, 2, 4, fx_root, "svm");
  cross_validator.Run();
  
  fx_done();
}

