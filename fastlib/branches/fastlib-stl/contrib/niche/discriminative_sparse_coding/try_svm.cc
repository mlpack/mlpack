
#include <fastlib/fastlib.h>



#include <mlpack/svm/svm.h>

#include <armadillo>
#include <fastlib/base/arma_compat.h>



using namespace arma;
using namespace std;

using std::string;
using std::vector;

int main(int argc, char *argv[]) {
  
  fx_module *root = fx_init(argc, argv, NULL);
  //srand(time(NULL));

  u32 n_dims = 50;
  u32 n_points = 10000;

  vec w = randn(n_dims, 1);
  mat data = randn(n_dims, n_points);
  for(u32 i = 0; i < n_points / 2; i++) {
    data.col(i) += 2 * w;
  }
  for(u32 i = n_points / 2; i < n_points; i++) {
    data.col(i) -= 2 * w;
  }
  
  mat svm_data(n_dims + 1, n_points);
  svm_data(span(0, n_dims - 1), span::all) = data;
  svm_data(n_dims, span(0, (n_points / 2) - 1)).fill(1.0);
  svm_data(n_dims, span(n_points / 2, n_points - 1)).fill(-1.0);
  
  printf("optimal dot products\n");
  for(u32 i = 0; i < n_points; i++) {
    printf("%f\n", dot(data.col(i), w));
  }
  
  
  Dataset trainset;
  trainset.CopyMatrix(svm_data);
  
  SVM<SVMLinearKernel> svm;
  
  u32 learner_typeid = 0;
  datanode* svm_module = fx_submodule(fx_root, "svm");
  fx_set_param_double(svm_module, "c", 0.1);
  
  svm.InitTrain(learner_typeid, trainset, svm_module);

  printf("predictions\n");
  u32 n_errors = 0;
  for(u32 i = 0; i < n_points; i++) {
    double y_hat = svm.Predict(0, data.col(i));
    if((i < n_points / 2) && (y_hat != 1.0)) {
      n_errors++;
    }
    else if((i >= n_points / 2) && (y_hat != -1.0)) {
      n_errors++;
    }
    printf("%f\n", y_hat);
  }
  
  printf("n_errors = %d\n", n_errors);
  
  fx_done(NULL);
}

