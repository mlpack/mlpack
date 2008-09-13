#include "fastlib/fastlib_int.h"
#include "fastlib/thor/thor.h"
#include "thor_kde.h"
#include "../series_expansion/kernel_aux.h"

int main(int argc, char *argv[]) {
  fx_init(argc, argv, NULL);
  ThorKde<GaussianKernelAux> thor_kde;
 
  // initialize and compute
  thor_kde.Init(fx_root);
  thor_kde.Compute(fx_root);

  fx_done(fx_root);
  return 0;
}
