#include "fastlib/fastlib_int.h"
#include "lpr.h"

int main(int argc, char *argv[]) {

  fx_init(argc, argv);

  NaiveLpr<GaussianKernel> naive_lpr;
  naive_lpr.Init(5);
  naive_lpr.Compute();

  fx_done();
  return 0;
}
