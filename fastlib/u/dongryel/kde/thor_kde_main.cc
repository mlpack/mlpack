#include "fastlib/fastlib_int.h"
#include "u/dongryel/series_expansion/kernel_aux.h"
#include "thor_kde.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  ThorKde<GaussianKernel, GaussianKernelAux> thor_kde_;
  fx_done();
  return 0;
}
