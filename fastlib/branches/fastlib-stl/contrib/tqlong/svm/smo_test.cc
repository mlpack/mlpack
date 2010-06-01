#include <fastlib/fastlib.h>
#include "svm.h"

using namespace SVMLib;

int main() {
  Matrix data;
  data::Load("test4.txt", &data);
  printf("n_cols = %d n_rows = %d\n", data.n_cols(), data.n_rows());

  Matrix X;
  Vector y;
  Vector box;
  KernelFunction linKernel(KernelFunction::LINEAR);
  KernelFunction quadKernel(KernelFunction::POLYNOMIAL, 2);
  KernelFunction rbfKernel(KernelFunction::RBF, 20);
  SMOOptions options;

  Vector shiftVec, scaleVec;
  SplitDataToXy(data, X, y);
  ScaleData(X, shiftVec, scaleVec);  
  //ot::Print(X);
  //ot::Print(shiftVec);
  //ot::Print(scaleVec);
  //ot::Print(y);
  SetBoxConstraint(y, 1e6, box);
  //ot::Print(box);
  options.verbose = SMOOptions::NONE;
  options.tolKKT = 1e-3;
  options.KKTViolationLevel = 0.005;

  Vector alpha; 
  IndexSet SVs(X.n_cols());
  double offset;
  Kernel kernel(quadKernel, X);
  seqminopt(X, y, box, kernel, options, alpha, SVs, offset);

  //ot::Print(alpha);
  printf("n_SVs = %d\n", SVs.get_n());
  SVs.print(alpha);
  printf("offset = %f\n", offset);

  printf("Total error = %d\n", svm_total_error(kernel, y, alpha, SVs, offset));

  return 0;
}
