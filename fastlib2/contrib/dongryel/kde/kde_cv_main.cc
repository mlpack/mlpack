#include "kde_cv.h"

int main(int argc, char *argv[]) {
  
  KdeCV<GaussianKernel> kde_cv_algorithm;
  
  kde_cv_algorithm.Init();

  return 0;
}
