#include "fastlib/fastlib_int.h"
#include "thor/thor.h"
#include "thor_kde.h"
#include "u/dongryel/series_expansion/kernel_aux.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv);
  ThorKde<GaussianKernel, GaussianKernelAux> thor_kde;
 
  // initialize and compute
  thor_kde.Init(fx_root);
  thor_kde.NaiveCompute();
  //thor_kde.Compute();
 
  fx_done();
  return 0;
}
